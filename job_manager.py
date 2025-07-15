#!/usr/bin/env python3
"""
Async Job Queue Manager for handling long-running transcription tasks.

This module provides:
1. Job queue for async processing
2. Job status tracking and persistence
3. Smart routing based on file duration
4. Progress reporting capabilities
"""

import asyncio
import json
import logging
import threading
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import concurrent.futures
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TranscriptionJob:
    """Transcription job data structure."""
    id: str
    file_path: str
    status: JobStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    options: Dict[str, Any] = None
    progress: float = 0.0
    estimated_duration: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert job to dictionary for JSON serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionJob':
        """Create job from dictionary."""
        data['status'] = JobStatus(data['status'])
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        if data.get('started_at'):
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if data.get('completed_at'):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


class JobManager:
    """Manages async transcription jobs with persistence."""
    
    def __init__(self, jobs_dir: Optional[Path] = None, max_workers: int = 2):
        """Initialize job manager.
        
        Args:
            jobs_dir: Directory for job persistence (default: ./jobs)
            max_workers: Maximum concurrent workers
        """
        self.jobs_dir = Path(jobs_dir or "./jobs")
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        
        self.jobs: Dict[str, TranscriptionJob] = {}
        self.job_queue = asyncio.Queue()
        self.workers = []
        self.max_workers = max_workers
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._lock = threading.Lock()
        
        # Load existing jobs
        self._load_jobs()
        
        # Callbacks
        self.transcribe_callback: Optional[Callable] = None
        
    def _load_jobs(self):
        """Load jobs from disk."""
        jobs_file = self.jobs_dir / "jobs.json"
        if jobs_file.exists():
            try:
                with open(jobs_file, 'r') as f:
                    jobs_data = json.load(f)
                    for job_data in jobs_data:
                        job = TranscriptionJob.from_dict(job_data)
                        self.jobs[job.id] = job
                logger.info(f"Loaded {len(self.jobs)} existing jobs")
            except Exception as e:
                logger.error(f"Failed to load jobs: {e}")
    
    def _save_jobs(self):
        """Save jobs to disk."""
        jobs_file = self.jobs_dir / "jobs.json"
        try:
            jobs_data = [job.to_dict() for job in self.jobs.values()]
            with open(jobs_file, 'w') as f:
                json.dump(jobs_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save jobs: {e}")
    
    async def start(self):
        """Start the job manager and workers."""
        if self._running:
            return
            
        self._running = True
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
        
        logger.info(f"Job manager started with {self.max_workers} workers")
    
    async def stop(self):
        """Stop the job manager."""
        self._running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        # Save final state
        self._save_jobs()
        
        logger.info("Job manager stopped")
    
    async def _worker(self, worker_id: str):
        """Worker coroutine for processing jobs."""
        logger.info(f"{worker_id} started")
        
        while self._running:
            try:
                # Get job from queue (with timeout to check running status)
                job_id = await asyncio.wait_for(
                    self.job_queue.get(), 
                    timeout=1.0
                )
                
                # Process job
                await self._process_job(job_id, worker_id)
                
            except asyncio.TimeoutError:
                # Check if we should continue
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"{worker_id} error: {e}")
        
        logger.info(f"{worker_id} stopped")
    
    async def _process_job(self, job_id: str, worker_id: str):
        """Process a single job."""
        job = self.jobs.get(job_id)
        if not job:
            logger.error(f"Job {job_id} not found")
            return
        
        logger.info(f"{worker_id} processing job {job_id}")
        
        # Update job status
        with self._lock:
            job.status = JobStatus.RUNNING
            job.started_at = datetime.now()
            self._save_jobs()
        
        try:
            # Call transcription callback
            if self.transcribe_callback:
                # Run transcription in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self.transcribe_callback,
                    job.file_path,
                    job.options or {}
                )
                
                # Update job with result
                with self._lock:
                    job.status = JobStatus.COMPLETED
                    job.completed_at = datetime.now()
                    job.result = result
                    job.progress = 100.0
                    self._save_jobs()
                
                logger.info(f"Job {job_id} completed successfully")
            else:
                raise ValueError("No transcribe callback set")
                
        except Exception as e:
            # Handle job failure
            with self._lock:
                job.status = JobStatus.FAILED
                job.completed_at = datetime.now()
                job.error = str(e)
                self._save_jobs()
            
            logger.error(f"Job {job_id} failed: {e}")
    
    def create_job(self, file_path: str, options: Dict[str, Any], 
                   estimated_duration: Optional[float] = None) -> str:
        """Create a new transcription job.
        
        Args:
            file_path: Path to audio/video file
            options: Transcription options
            estimated_duration: Estimated audio duration in seconds
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        job = TranscriptionJob(
            id=job_id,
            file_path=file_path,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            options=options,
            estimated_duration=estimated_duration
        )
        
        with self._lock:
            self.jobs[job_id] = job
            self._save_jobs()
        
        # Add to queue
        asyncio.create_task(self.job_queue.put(job_id))
        
        logger.info(f"Created job {job_id} for {file_path}")
        return job_id
    
    def get_job(self, job_id: str) -> Optional[TranscriptionJob]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    def list_jobs(self, status: Optional[JobStatus] = None, 
                  limit: int = 100) -> List[TranscriptionJob]:
        """List jobs with optional status filter."""
        jobs = list(self.jobs.values())
        
        # Filter by status
        if status:
            jobs = [j for j in jobs if j.status == status]
        
        # Sort by created time (newest first)
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        
        # Apply limit
        return jobs[:limit]
    
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or running job."""
        job = self.jobs.get(job_id)
        if not job:
            return False
        
        if job.status in [JobStatus.PENDING, JobStatus.RUNNING]:
            with self._lock:
                job.status = JobStatus.CANCELLED
                job.completed_at = datetime.now()
                self._save_jobs()
            return True
        
        return False
    
    def clean_old_jobs(self, days: int = 7):
        """Remove completed/failed jobs older than specified days."""
        cutoff = datetime.now().timestamp() - (days * 24 * 60 * 60)
        
        with self._lock:
            # Find jobs to remove
            to_remove = []
            for job_id, job in self.jobs.items():
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
                    if job.completed_at and job.completed_at.timestamp() < cutoff:
                        to_remove.append(job_id)
            
            # Remove old jobs
            for job_id in to_remove:
                del self.jobs[job_id]
            
            if to_remove:
                self._save_jobs()
                logger.info(f"Cleaned {len(to_remove)} old jobs")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get job queue statistics."""
        stats = {
            "total_jobs": len(self.jobs),
            "pending": sum(1 for j in self.jobs.values() if j.status == JobStatus.PENDING),
            "running": sum(1 for j in self.jobs.values() if j.status == JobStatus.RUNNING),
            "completed": sum(1 for j in self.jobs.values() if j.status == JobStatus.COMPLETED),
            "failed": sum(1 for j in self.jobs.values() if j.status == JobStatus.FAILED),
            "cancelled": sum(1 for j in self.jobs.values() if j.status == JobStatus.CANCELLED),
            "queue_size": self.job_queue.qsize(),
            "workers": self.max_workers
        }
        return stats


class SmartRouter:
    """Smart routing logic for sync vs async processing."""
    
    # Thresholds for routing decisions
    ASYNC_DURATION_THRESHOLD = 300  # 5 minutes
    ASYNC_FILE_SIZE_THRESHOLD = 100 * 1024 * 1024  # 100MB
    
    @staticmethod
    def should_use_async(file_path: str, duration: Optional[float] = None,
                        file_size: Optional[int] = None) -> bool:
        """Determine if file should be processed asynchronously.
        
        Args:
            file_path: Path to audio/video file
            duration: Audio duration in seconds (if known)
            file_size: File size in bytes (if known)
            
        Returns:
            True if async processing recommended
        """
        # If duration is provided and exceeds threshold
        if duration and duration > SmartRouter.ASYNC_DURATION_THRESHOLD:
            return True
        
        # If file size is provided and exceeds threshold
        if file_size and file_size > SmartRouter.ASYNC_FILE_SIZE_THRESHOLD:
            return True
        
        # Check actual file size if not provided
        if not file_size:
            try:
                file_size = Path(file_path).stat().st_size
                if file_size > SmartRouter.ASYNC_FILE_SIZE_THRESHOLD:
                    return True
            except:
                pass
        
        # Default to sync for smaller files
        return False
    
    @staticmethod
    def estimate_processing_time(duration: float, model: str = "large-v3-turbo") -> float:
        """Estimate processing time based on duration and model.
        
        Args:
            duration: Audio duration in seconds
            model: Whisper model name
            
        Returns:
            Estimated processing time in seconds
        """
        # Speed multipliers based on model (approximate)
        speed_multipliers = {
            "tiny": 0.1,
            "base": 0.15,
            "small": 0.25,
            "medium": 0.4,
            "large": 0.6,
            "large-v3": 0.6,
            "large-v3-turbo": 0.3
        }
        
        # Get multiplier for model
        multiplier = 0.5  # Default
        for key, value in speed_multipliers.items():
            if key in model.lower():
                multiplier = value
                break
        
        # Estimate processing time
        return duration * multiplier