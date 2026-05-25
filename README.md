# AI Coding Agent

Production-ready AI Coding Agent CLI — an autonomous development companion that understands repositories, writes code, executes commands, debugs issues, and learns from experience.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Layer (Typer/Rich)                │
├─────────────────────────────────────────────────────────┤
│                    API Layer (FastAPI)                   │
├─────────────────────────────────────────────────────────┤
│              Agent Orchestration Engine                  │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │ Planner │ │ Executor │ │Reflector │ │State Mgmt │  │
│  └─────────┘ └──────────┘ └──────────┘ └───────────┘  │
├─────────────────────────────────────────────────────────┤
│  Multi-Agent System (Manager/Coder/Reviewer/Debugger)   │
├──────────┬──────────┬───────────┬───────────────────────┤
│  Tools   │  Memory  │  Models   │  Repository Engine    │
│  System  │  System  │  Layer    │  Code Edit Engine     │
├──────────┴──────────┴───────────┴───────────────────────┤
│              Core (Config/Events/DI/Base)                │
├─────────────────────────────────────────────────────────┤
│         Infrastructure (DB/Redis/Vector/Docker)          │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone and install
git clone <repo-url> && cd ai-agent
pip install uv
uv pip install -e .

# Copy environment config
cp .env.example .env
# Edit .env with your API keys
```

### Docker (Full Stack)

```bash
cd docker
docker compose up -d
```

This starts: PostgreSQL, Redis, ChromaDB, Ollama, and the agent.

### Basic Usage

```bash
# Interactive chat
ai chat

# Execute a task autonomously
ai run "Add error handling to the API endpoints in src/api/"

# Create a plan
ai plan "Refactor the authentication system to use JWT"

# Analyze a repository
ai init .

# Multi-agent collaboration
ai agents "Build a REST API for user management" --strategy parallel

# Check system health
ai doctor
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `ai chat` | Interactive chat with slash commands |
| `ai run <task>` | Execute a task autonomously |
| `ai plan <task>` | Create an execution plan |
| `ai init [path]` | Analyze repository structure |
| `ai memory` | Manage agent memory |
| `ai tools` | List available tools |
| `ai models` | List available models |
| `ai doctor` | System health check |
| `ai agents <task>` | Multi-agent collaboration |
| `ai train` | Training pipeline |
| `ai benchmark` | Run benchmarks |
| `ai logs` | View agent logs |

## Features

### Agent Engine
- Step-by-step reasoning with planning and reflection
- Automatic retry with replanning on failure
- Self-evaluation and quality scoring
- Token-optimized context management

### Tool System
- **Shell**: Execute commands with timeout and output capture
- **Filesystem**: Read, write, edit files with backup
- **Git**: Full git operations (status, diff, commit, branch)
- **Code Search**: AST-aware symbol search across codebases
- **Python Exec**: Isolated Python execution
- **Docker Sandbox**: Full container isolation for untrusted code
- **HTTP**: API requests

### Memory System
- Short-term memory with automatic promotion
- Long-term semantic memory with embeddings
- Episodic memory for task execution history
- Vector similarity search with scoring and decay

### Multi-Agent System
- **Manager**: Task decomposition and delegation
- **Coder**: Production code implementation
- **Reviewer**: Code review and quality checks
- **Debugger**: Error analysis and fixes
- **Researcher**: Codebase investigation
- **Planner**: Project planning and milestones

### Model Support
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude Sonnet 4)
- Groq (Llama 3)
- Ollama (local models)
- vLLM (self-hosted)
- OpenRouter (any model)
- Automatic fallback between providers

### Training Pipeline
- Synthetic dataset generation
- LoRA/QLoRA fine-tuning
- Self-play training
- Evaluation benchmarks

### Self-Improvement
- Deep reflection on task execution
- Multi-criteria output scoring
- Prompt optimization
- Failure pattern analysis
- Workflow improvement suggestions

## Configuration

Environment variables (`.env`):

```bash
# Required: At least one LLM provider key
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Or use local models
OLLAMA_BASE_URL=http://localhost:11434

# Agent settings
AI_AGENT_DEFAULT_MODEL=anthropic/claude-sonnet-4-20250514
AI_AGENT_MAX_ITERATIONS=50
AI_AGENT_TIMEOUT=300
```

## API Server

```bash
# Start the API server
uvicorn ai_agent.api:app --host 0.0.0.0 --port 8080

# Or use Docker
docker compose up api
```

Endpoints:
- `GET /health` - Health check
- `POST /chat` - Synchronous chat
- `POST /tasks` - Async task execution
- `GET /tasks/{id}` - Task status
- `POST /agents/multi` - Multi-agent
- `POST /memory/store` - Store memory
- `POST /memory/search` - Search memory
- `GET /models` - List models
- `GET /tools` - List tools

## Project Structure

```
src/ai_agent/
├── core/          # Config, events, DI, base types
├── agents/        # Agent engine, multi-agent orchestration
├── tools/         # Shell, filesystem, git, code search, sandbox
├── memory/        # Memory manager, vector store, embeddings
├── models/        # LLM client, model registry
├── repository/    # Repo analysis, framework detection
├── editing/       # Code editor, diff/patch, rollback
├── training/      # Dataset gen, fine-tuning, evaluation
├── evals/         # Self-improvement, reflection, scoring
├── cli/           # Typer CLI, interactive session
├── api/           # FastAPI REST API
└── __init__.py
```

## Development

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Lint
ruff check src/

# Type check
mypy src/
```

## Deployment

### Production Docker

```bash
# Build
docker build -t ai-agent --target api -f docker/Dockerfile .

# Run
docker run -d --env-file .env -p 8080:8080 ai-agent
```

### Scaling Strategy

- **Horizontal**: Multiple API instances behind a load balancer
- **Vertical**: GPU instances for local model inference
- **Queue-based**: Redis task queue for async execution
- **Caching**: Redis for LLM response caching
- **Vector DB**: ChromaDB cluster for memory at scale

## License

MIT
# ai-agent-train
