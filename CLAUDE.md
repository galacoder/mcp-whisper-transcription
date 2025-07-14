# CLAUDE.md - Concise Project Guide for Claude Code

This file provides essential guidance for Claude Code (Sonnet & Opus 4) when working with this repository.

## 🏗️ Project Architecture

MCP Server for Whisper Transcription:
- **Main Server**: FastMCP-based server (`src/whisper_mcp_server.py`)
- **Core Transcriber**: MLX Whisper implementation (`transcribe_mlx.py`)
- **Utilities**: Audio processing, formatting (`whisper_utils.py`)
- **Tests**: Comprehensive test suite (`test_*.py`)

## 🧠 Memory Strategy

### Dual-Memory System
- **Local (OpenMemory)**: Project-specific knowledge, quick access
- **Global (Graphiti)**: Cross-project patterns, reusable solutions

### Quick Reference
- **Local**: `search_memory("[query]")` - This project's specifics
- **Global**: `mcp__graphiti-memory__search_memory_nodes --query="[pattern]" --group_ids=["global_patterns"]`

### What Goes Where
**Local Memory** (This Project):
- MCP server patterns, FastMCP usage
- Whisper transcription workflows
- Audio processing techniques
- Project-specific configurations

**Global Memory** (Your Knowledge Base):
- MCP protocol patterns
- FastMCP architectural patterns
- Audio/video processing patterns
- Python async patterns

## 🚀 Mandatory Task Workflows

### Task Start Checklist
ALWAYS execute in order:
1. Check memories:
   - Local: `search_memory("task_context [task_id]")`
   - Global: `mcp__graphiti-memory__search_memory_nodes --query="[technology] patterns" --group_ids=["global_patterns"]`
2. `mcp__task-master__get_task --id="[task_id]" --projectRoot="/Users/sangle/Dev/action/projects/mcp-servers/tools/mcp-whisper-transcription"`
3. Check/create tag context with `mcp__task-master__list_tags`
4. If unfamiliar tech:
   - Use Context7: `mcp__context7__resolve-library-id --libraryName="[technology]"` then `mcp__context7__get-library-docs --context7CompatibleLibraryID="[resolved_id]"`
   - Web search: `WebSearch --query="[topic] best practices implementation"`
   - Save findings: `mcp__openmemory__add_memories --text="Research: [findings]"`
5. If no subtasks: `mcp__task-master__analyze_project_complexity --ids="[task_id]"`
6. `mcp__task-master__set_task_status --id="[task_id]" --status="in-progress"`
7. Create TodoWrite for task breakdown
8. Search implementation patterns in both memories

### Task Completion Workflow
Execute ALL before marking complete:
1. `mcp__task-master__update_task --id="[task_id]" --prompt="COMPLETED: [summary]"`
2. If research conducted: Save findings to memory `mcp__openmemory__add_memories --text="Task [task_id] research: [findings]"`
3. Update memories:
   - Local: `mcp__openmemory__add_memories --text="Task [task_id]: [specific_solution]"`
   - Global (if reusable): `mcp__graphiti-memory__add_memory --name="Pattern: [concept]" --episode_body="[details]" --source="text" --group_id="global_patterns"`
4. Run quality checks (see below)
5. Create comprehensive commit
6. `mcp__task-master__set_task_status --id="[task_id]" --status="completed"`
7. Archive: `mcp__openmemory__add_memories --text="ARCHIVED: Task [task_id] complete"`

## 🔨 Essential Commands

### Build Commands
```bash
# Install dependencies
poetry install

# Run server in development
python -m src.whisper_mcp_server

# Run tests
poetry run pytest
poetry run pytest -v  # verbose
poetry run pytest test_resource_endpoints.py  # specific test

# Code quality
poetry run black .  # format code
poetry run flake8 src tests  # lint
poetry run mypy src  # type checking
```

### Quality Checks
```bash
# All checks
poetry run pytest && poetry run flake8 src tests && poetry run mypy src

# Before commit
poetry run black . && poetry run pytest
```

Always run before marking task complete.

## 💡 Development Principles

### Top-Down Approach
All implementations:
1. Main entry point first (main function or primary handler)
2. Workflow comments (# 1. Step, # 2. Step)
3. Break complexity into smaller functions
4. Implementation details last

### TDD Workflow
NEVER write implementation before tests:
1. **RED**: Write failing test defining functionality
2. **GREEN**: Minimal code to pass test
3. **REFACTOR**: Clean up keeping tests green

### FastMCP Patterns
```python
# Tool definition
@mcp.tool
async def tool_name(param: str) -> dict:
    """Tool description."""
    # Implementation

# Resource definition
@mcp.resource("protocol://path")
async def resource_name() -> dict:
    """Resource description."""
    # Implementation

# Resource with parameters
@mcp.resource("protocol://path/{param}")
async def resource_with_param(param: str) -> dict:
    """Resource with parameter."""
    # Implementation
```

## 📁 Project Structure

```
mcp-whisper-transcription/
├── src/
│   └── whisper_mcp_server.py  # Main MCP server
├── transcribe_mlx.py           # Core transcriber
├── whisper_utils.py            # Utilities
├── test_*.py                   # Test files
├── logs/                       # Log directory
│   └── transcription_history.json  # History tracking
└── temp/                       # Temporary files
```

## 🎯 Performance & Quality

### Performance Targets
- Transcription speed: >2x realtime (model dependent)
- Server startup: <5s
- Resource response: <100ms
- Test coverage: ≥85%
- Memory usage: <500MB baseline

### MCP Resource Patterns
Resources should:
- Return valid JSON always
- Handle missing data gracefully
- Support filtering/pagination where applicable
- Include metadata in responses
- Use proper URI schemes

## 📝 Git & Documentation

### Commit Message Template
```
<type>(scope): <brief description>

<detailed description including:>
• Core features implemented
• Performance metrics achieved
• Technical approach used
• Files changed and purpose
• Test/demo outcomes

Ready for <next task>.
```

Types: feat, fix, refactor, test, docs, perf  
**NO Claude Code attribution lines**

### Tag Management
Tags isolate task contexts per branch:
- Auto-create: `mcp__task-master__add_tag --fromBranch=true --projectRoot="/Users/sangle/Dev/action/projects/mcp-servers/tools/mcp-whisper-transcription"`
- List: `mcp__task-master__list_tags --showMetadata=true`  
- Switch: `mcp__task-master__use_tag --name="[tag]"`
- Copy: `mcp__task-master__copy_tag --sourceName="[src]" --targetName="[dst]"`
- Clean up: `mcp__task-master__delete_tag --name="[tag]"`

## 🆘 Emergency Help

- Stuck? → `search_memory("common_errors")`
- Need patterns? → `search_memory("working_patterns")`
- MCP patterns? → `mcp__graphiti-memory__search_memory_nodes --query="mcp patterns" --group_ids=["global_patterns"]`
- FastMCP docs? → `mcp__context7__resolve-library-id --libraryName="fastmcp"`
- Task unclear? → Review task-master details
- Complex task? → `analyze_project_complexity`
- Wrong context? → `list_tags` then `use_tag`

### Knowledge Promotion
When you discover a reusable pattern:
1. Document in local memory for immediate use
2. Extract general pattern → Add to global Graphiti
3. Future projects benefit from this knowledge

Example: `mcp__graphiti-memory__add_memory --name="Pattern: MCP Resource Implementation" --episode_body="Resource endpoints with FastMCP using @mcp.resource decorator..." --source="text" --group_id="global_patterns"`

## 🔑 Memory Search Keys (Local Project Memory)

*For global patterns, use: `mcp__graphiti-memory__search_memory_nodes --query="[pattern]" --group_ids=["global_patterns"]`*

### Project-Specific Memories
- `mcp_server_patterns` - MCP server implementation patterns
- `fastmcp_usage` - FastMCP-specific patterns
- `whisper_transcription` - Transcription workflows
- `audio_processing` - Audio/video handling
- `resource_endpoints` - MCP resource patterns
- `test_patterns` - Testing approaches

### Workflow Memories
- `task_start_checklist` - Mandatory startup sequence
- `task_completion_workflow` - Completion steps
- `commit_message_template` - Git standards
- `tag_management` - Context switching
- `quality_checks` - Validation commands
- `performance_targets` - System requirements
- `research_commands` - AI research
- `complexity_analysis` - Task expansion
- `emergency_help` - Troubleshooting

**Usage Tip**: Always start with summary files, then load specific patterns as needed to minimize context usage.

---
**Remember**: This workflow is MANDATORY, not optional. Every step must be completed for successful task execution.