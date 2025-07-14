# CLAUDE.md - Core Rules with Mandatory Workflows

## 🚀 MANDATORY Workflow for ALL Tasks

### Before Starting ANY Task
**ALWAYS execute these steps in order:**

1. **Search OpenMemory for Context**: 
   ```bash
   search_memory("task_context [task_id]")
   search_memory("task_start_checklist")
   ```

2. **Get Task Details**: 
   ```bash
   mcp__task-master__get_task --id="[task_id]" --projectRoot="/Users/sangle/Dev/action/projects/prox"
   ```

3. **Set Task Status**: 
   ```bash
   mcp__task-master__set_task_status --id="[task_id]" --status="in-progress" --projectRoot="/Users/sangle/Dev/action/projects/prox"
   ```

4. **Create TodoWrite** for task breakdown

5. **Search for Implementation Patterns**:
   - Python: `search_memory("python_patterns")`
   - React: `search_memory("react_patterns")`
   - TDD: `search_memory("tdd_red_green_refactor")`
   - Top-down: `search_memory("topdown_approach_examples")`

### During Implementation (MANDATORY)

1. **Follow TDD Methodology**:
   - RED: Write failing test first
   - GREEN: Minimal implementation to pass
   - REFACTOR: Clean up while keeping tests green
   - NEVER write implementation before tests

2. **Follow Top-Down Structure**:
   - Start with main() function or primary component
   - Add workflow comments (# 1. Step one, # 2. Step two)
   - Implement supporting functions after main workflow
   - Use type hints and proper error handling

3. **Store Decisions in Memory**:
   ```bash
   mcp__openmemory__add_memories --text="[architectural decision or pattern]"
   ```

### Upon Task Completion (MANDATORY)

**Execute ALL steps before considering task complete:**

1. **Update Task with Summary**:
   ```bash
   mcp__task-master__update_task --id="[task_id]" --prompt="COMPLETED: [detailed summary]" --projectRoot="/Users/sangle/Dev/action/projects/prox"
   ```

2. **Store Knowledge in OpenMemory**:
   ```bash
   mcp__openmemory__add_memories --text="Task [task_id] completed: [key learnings and implementation details]"
   ```

3. **Run Quality Checks**:
   ```bash
   # Backend
   cd apps/backend && poetry run pytest && poetry run flake8 src
   
   # Frontend
   cd apps/consumer && pnpm test && pnpm lint
   ```

4. **Create Comprehensive Commit**:
   ```bash
   search_memory("commit_message_template")
   # Then create commit following template
   ```

5. **Mark Task Complete**:
   ```bash
   mcp__task-master__set_task_status --id="[task_id]" --status="completed" --projectRoot="/Users/sangle/Dev/action/projects/prox"
   ```

6. **Archive Implementation**:
   ```bash
   mcp__openmemory__add_memories --text="ARCHIVED: Task [task_id] implementation complete with [files changed, patterns used, outcomes achieved]"
   ```

## 🧠 Memory Pattern Reference

**Primary**: OpenMemory search commands  
**Fallback**: Local patterns in `.claude/patterns.md`

| Pattern Type | Search Command | Fallback Section |
|--------------|----------------|------------------|
| Task workflow | `search_memory("task_start_checklist")` | TASK_START_CHECKLIST |
| Top-down examples | `search_memory("topdown_approach_examples")` | TOPDOWN_APPROACH_EXAMPLES |
| TDD workflow | `search_memory("tdd_red_green_refactor")` | TDD_RED_GREEN_REFACTOR_WORKFLOW |
| Commit templates | `search_memory("commit_message_template")` | COMMIT_MESSAGE_TEMPLATE |
| Python patterns | `search_memory("python_patterns")` | PYTHON_PATTERNS |
| React patterns | `search_memory("react_patterns")` | REACT_PATTERNS |
| Plugin architecture | `search_memory("plugin_system_patterns")` | PLUGIN_SYSTEM_PATTERNS |

**If OpenMemory fails**: Read `.claude/patterns.md` for all pattern details

## 🔨 Build Commands

### Monorepo
- `pnpm dev` - Run all apps
- `pnpm build` - Build all apps  
- `pnpm lint` - Lint all code

### Backend
- `cd apps/backend && poetry run pytest` - Run tests
- `cd apps/backend && poetry run pytest --cov` - With coverage
- `cd apps/backend && poetry run flake8 src` - Lint Python code

### Frontend
- `cd apps/consumer && pnpm test` - Unit tests
- `cd apps/consumer && pnpm test:e2e` - E2E tests
- `cd apps/consumer && pnpm lint` - Lint TypeScript
- `cd apps/consumer && pnpm type-check` - Type checking

## 📁 Project Structure
- **Backend**: FastAPI + PostgreSQL (apps/backend/)
- **Consumer**: Next.js 15 (apps/consumer/)
- **Producer**: Next.js 14 (apps/producer/)
- **Docs**: Mintlify (apps/docs/)

## 🎯 Performance Targets
- Plugin activation: <10ms
- API response: <100ms
- Test coverage: ≥85%
- Memory usage: <50MB per plugin

## ⚠️ NON-NEGOTIABLE Rules

1. **NEVER skip memory search** - Always start with `search_memory("task_start_checklist")`
2. **NEVER skip TDD** - Tests must come before implementation
3. **NEVER skip top-down approach** - Always start with main() workflow
4. **NEVER skip quality checks** - Tests and linting required before completion
5. **NEVER skip comprehensive commits** - Use template from memory
6. **NEVER skip memory archival** - Store learnings for future tasks

## 🔧 Custom Commands Available

- `/start-task [task_id]` - Execute complete task startup workflow
- `/tdd-workflow [feature]` - Follow TDD implementation steps
- `/create-commit [task_id]` - Create comprehensive commit

## 💡 Emergency Troubleshooting

- Stuck? → `search_memory("common_errors")`
- Need patterns? → `search_memory("working_patterns")`
- Architecture questions? → Check `specs/workflow/`
- Task unclear? → Review task-master details and memory context

---
**⚡ REMEMBER: This workflow is MANDATORY, not optional. Every step must be completed for successful task execution.**