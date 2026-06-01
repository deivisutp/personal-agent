# Personal Agents

AI-powered agents for technical refinement and pattern compliance, running locally with Ollama.

## Pending: Analise de impacto agent.

## Roadmap Status
- [x] **Phase 1**: Foundation (Ollama client, base agent, ChromaDB)
- [x] **Phase 2**: PR Review Agent + GitHub MCP integration
- [x] **Phase 3**: Pattern Agent + Azure DevOps MCP integration
- [x] **Phase 4**: API layer, error handling, deployment
- [x] **Phase 5-7**: Dev Assistant Agent (RAG over internal docs) + persistent chat sessions + HTMX web chat UI
- [x] **Phase 8**: Hybrid retrieval (BM25 + vector via RRF), LLM re-rank, SSE streaming, knowledge browser UI

## Agents

### 1. PR Review Agent
Reviews Pull Requests for code quality, best practices, security, and performance.
- **Input**: PR diffs, file changes
- **Output**: Structured review with prioritized feedback
- **Integration**: GitHub MCP (Phase 2)

### 2. Pattern & NFR Agent
Evaluates features against company patterns and generates non-functional requirements.
- **Input**: Feature descriptions, user stories
- **Output**: Pattern compliance analysis, generated NFRs
- **Integration**: Azure DevOps MCP (Phase 3)
- **RAG**: ChromaDB for pattern document storage

### 3. Dev Assistant Agent
Conversational assistant grounded in YOUR internal engineering documentation
(backend structure, business rules, database modeling, PL/SQL objects, frontend
patterns). Answers questions like *"How do I implement a grid WCPanelAction?"*
using only what's in the knowledge base, with citations.

- **Knowledge base**: dedicated Chroma collection `dev_knowledge` (separate from NFR patterns).
- **Smart ingestion**: markdown-section-aware splitter that preserves code fences,
  strips wiki noise (`[[_TOC_]]`, HTML comments, image-only lines), and tags chunks
  with `doc_type`, `layer`, `language`, `heading_path` metadata for filtered retrieval.
- **Sources**: Azure DevOps wiki, local folders, single files, raw text — declared in
  `knowledge_manifest.yaml`.
- **Chat memory**: SQLite-backed sessions (`data/chat.db`) so conversations persist.
- **Web UI**: HTMX single-page chat at `/dev-assistant/ui` with markdown + syntax
  highlighting and a "Sources" panel under each answer.

#### Quick start

1. Copy and edit the manifest:

   ```bash
   copy knowledge_manifest.example.yaml knowledge_manifest.yaml
   # edit to point at your wiki paths / local folders
   ```

2. Dry-run to inspect what will be ingested (no embeddings yet):

   ```bash
   python scripts/prepare_knowledge.py knowledge_manifest.yaml --dry-run
   ```

3. Ingest for real:

   ```bash
   python scripts/prepare_knowledge.py knowledge_manifest.yaml
   # or via API:  POST /dev-assistant/ingest/manifest
   ```

4. Start the API and open the chat UI:

   ```bash
   agents-api
   # browse to http://localhost:8000/dev-assistant/ui
   ```

#### Dev Assistant API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dev-assistant/sessions` | GET / POST | List or create chat sessions |
| `/dev-assistant/sessions/{id}` | GET / DELETE | Get full message history / delete |
| `/dev-assistant/sessions/{id}/messages` | POST | Ask a question, get answer + sources |
| `/dev-assistant/stats` | GET | Knowledge base size |
| `/dev-assistant/ingest/text` | POST | Ingest a raw markdown blob |
| `/dev-assistant/ingest/path` | POST | Ingest a local file or directory |
| `/dev-assistant/ingest/wiki` | POST | Ingest an Azure DevOps wiki path |
| `/dev-assistant/ingest/manifest` | POST | Ingest from a manifest (path or inline) |
| `/dev-assistant/knowledge` | DELETE | Clear the dev knowledge collection |
| `/dev-assistant/ui` | GET | HTMX chat interface |

#### Filtered retrieval

Both API and UI accept `filters` (e.g. `{"doc_type": "frontend_pattern", "layer": "frontend"}`)
to scope the RAG search to specific document types — useful when the same term
exists in backend and frontend documentation.

#### Retrieval pipeline

For every question the agent runs:

1. **Hybrid candidate generation** — BM25 (lexical, identifier-aware tokenizer that
   splits `WCPanelAction` → `wc`, `panel`, `action`, `wcpanelaction`) **and**
   dense vector search are both run, fused with **Reciprocal Rank Fusion (RRF)**
   into a top-20 candidate set. Big quality win for symbol-heavy queries.
2. **LLM re-rank** — a fast prompt asks the model to pick the 5 most relevant
   chunks from the 20 candidates (returns a JSON list of indices). Falls back
   gracefully on parse errors. Toggle with `DevAssistantAgent(rerank=False)`.
3. **Answer generation** — top-5 chunks are inlined into the system prompt with
   `doc_type`, `layer`, `heading_path`, `source` headers; the model is instructed
   to cite them in a final `Sources:` section.

The BM25 index is built lazily from the Chroma collection and invalidated
automatically after every ingestion.

#### Streaming chat (SSE)

The web UI uses `EventSource` against:

```
GET /dev-assistant/sessions/{session_id}/messages/stream?question=...&doc_type=...&layer=...
```

It emits four event types:
- `sources` — the selected chunks (sent before the first token).
- `delta` — content chunks as they arrive from the model.
- `done` — final message id and full content (also persisted to SQLite).
- `error` — any stream-time failure.

#### Knowledge browser

The chat sidebar has a **Browse knowledge →** link that opens
`/dev-assistant/ui/knowledge`: paginated, filterable list of every indexed chunk
(metadata + preview) with a click-to-inspect detail pane showing the full
content and all metadata. Backed by:
- `GET /dev-assistant/knowledge/list?offset=&limit=&doc_type=&layer=`
- `GET /dev-assistant/knowledge/{chunk_id}`

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) running locally
- Mistral-Nemo model: `ollama pull mistral-nemo`
- (Optional) Embedding model: `ollama pull nomic-embed-text`

## Installation

```bash
# Clone the repository
cd c:\dev\personal-agents

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -e .

venv\Scripts\python.exe -m pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and configure:

```bash
copy .env.example .env
```

Edit `.env` with your settings:
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=mistral-nemo
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
```

## Usage

### PR Review Agent

```python
from agents.pr_review import PRReviewAgent
from agents.core.base_agent import AgentContext

agent = PRReviewAgent()

# Quick review
result = agent.quick_review("""
- def calculate(x):
-     return x * 2
+ def calculate(x: int) -> int:
+     return x * 2
""")
print(result)

# Full review with context
context = AgentContext(
    user_input="Review this PR",
    metadata={
        "pr_title": "Add type hints",
        "pr_description": "Added type hints to calculate function",
        "pr_diff": "...",
        "files_changed": ["utils.py"],
    }
)
result = agent.execute(context)
print(result.output)
```

### Pattern & NFR Agent

```python
from agents.pattern import PatternAgent
from pathlib import Path

agent = PatternAgent()

# Ingest company patterns
agent.ingest_patterns(Path("./docs/patterns"))

# Or ingest raw text
agent.ingest_text("""
## API Design Pattern
All REST APIs must:
- Use JSON for request/response bodies
- Include correlation IDs in headers
- Return standard error format
""", source_name="api_patterns")

# Generate NFRs for a feature
nfrs = agent.generate_nfrs(
    "User authentication system with OAuth2 support",
    categories=["Security", "Performance", "Reliability"]
)
print(nfrs)
```

### CLI Usage

```bash
# Run PR Review Agent interactively (paste diffs)
pr-agent

# Run GitHub PR Review (fetch PRs from GitHub)
github-review

# Run Pattern Agent interactively
pattern-agent

# Or via Python
python -m agents.cli          # PR Agent
python -m agents.cli github   # GitHub PR Review
python -m agents.cli pattern  # Pattern Agent
```

### GitHub PR Review

```python
import asyncio
from agents.pr_review import PRReviewAgent

async def review_pr():
    agent = PRReviewAgent()
    
    # Review a PR from GitHub
    result = await agent.review_github_pr(
        owner="microsoft",
        repo="vscode",
        pr_number=12345,
        post_comment=False,  # Set True to post review to GitHub
    )
    
    print(result.output)
    await agent.close()

asyncio.run(review_pr())
```

### Azure DevOps Feature Evaluation

```python
import asyncio
from agents.pattern import PatternAgent

async def evaluate_feature():
    agent = PatternAgent()
    
    # Ingest patterns from wiki
    await agent.ingest_wiki_patterns(
        wiki_name="Architecture",
        path="/Patterns",
    )
    
    # Evaluate a feature from Azure DevOps
    result = await agent.evaluate_azure_feature(
        feature_id=12345,
        include_user_stories=True,
    )
    
    print(result.output)
    
    # Generate NFRs and optionally create work items
    nfrs = await agent.generate_nfrs_for_feature(
        feature_id=12345,
        categories=["Security", "Performance"],
        create_work_items=False,  # Set True to create in Azure DevOps
    )
    
    print(nfrs.output)
    await agent.close()

asyncio.run(evaluate_feature())
```

## REST API

The agents are also available via a REST API built with FastAPI.

### Running the API

```bash
# Using the CLI command
agents-api

# Or with uvicorn directly
uvicorn agents.api.app:app --reload --port 8000

# Or with Docker
docker-compose up
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/health/ready` | GET | Readiness check |
| `/health/live` | GET | Liveness check |
| `/pr-review/review` | POST | Review a PR diff |
| `/pr-review/github` | POST | Review a GitHub PR via MCP |
| `/pattern/evaluate` | POST | Evaluate a feature |
| `/pattern/nfrs` | POST | Generate NFRs |
| `/pattern/azure/evaluate` | POST | Evaluate Azure DevOps feature |
| `/pattern/azure/nfrs` | POST | Generate NFRs for Azure DevOps feature |
| `/pattern/ingest` | POST | Ingest pattern content |

### Example API Calls

```bash
# Health check
curl http://localhost:8000/health

# Review a diff
curl -X POST http://localhost:8000/pr-review/review \
  -H "Content-Type: application/json" \
  -d '{"diff": "- old\n+ new", "title": "Fix bug"}'

# Generate NFRs
curl -X POST http://localhost:8000/pattern/nfrs \
  -H "Content-Type: application/json" \
  -d '{"feature_description": "User authentication service"}'
```

### API Documentation

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

## Project Structure

```
personal-agents/
├── src/
│   └── agents/
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py       # Configuration management
│       │   ├── llm_client.py   # Ollama wrapper
│       │   └── base_agent.py   # Base agent class
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── vector_store.py # ChromaDB integration
│       │   └── document_loader.py
│       ├── pr_review/
│       │   ├── __init__.py
│       │   └── agent.py        # PR Review Agent
│       ├── pattern/
│       │   ├── __init__.py
│       │   └── agent.py        # Pattern Agent
│       └── cli.py              # CLI entry points
├── tests/
│   ├── test_llm_client.py
│   └── test_agents.py
├── data/
│   └── chroma/                 # Vector DB storage
├── pyproject.toml
├── requirements.txt
├── .env.example
└── README.md
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src tests
ruff check src tests --fix

# Type checking
mypy src
```

## Roadmap

- [x] **Phase 1**: Foundation (Ollama client, base agent, ChromaDB)
- [ ] **Phase 2**: PR Review Agent + GitHub MCP integration
- [ ] **Phase 3**: Pattern Agent + Azure DevOps MCP integration
- [ ] **Phase 4**: API layer, error handling, deployment


## Use case usage
Here are the step-by-step use cases and instructions for using your new API and agents now that the server is running on http://127.0.0.1:8000.

There are two main ways to use the agents:

Through the REST API (useful for integrating into webhooks, CI/CD, or custom scripts).
Through the CLI (useful for local interactive use).
Use Case 1: PR Code Review (Via API)
Scenario: You have a git diff that you want to review for security, performance, or general best practices before committing.

Open another terminal or use Postman/curl.
Send a POST request to /pr-review/review:

```bash
curl -X POST http://localhost:8000/pr-review/review \
  -H "Content-Type: application/json" \
  -d '{
    "diff": "def calculate_total(items):\n    sum = 0\n    for i in items:\n        sum += i.price\n    return sum",
    "title": "Add calculate total function",
    "focus": "performance"
  }'
```
Expected Result: The API will return a structured JSON response containing the LLM's review, suggesting improvements (like using sum(i.price for i in items)).

Use Case 2: Evaluate a Feature against Architecture Patterns (Via API)
Scenario: You are writing a new microservice and want to check if your feature description aligns with the company's indexed architecture patterns.

First, make sure you have some patterns indexed (you can see in your logs Patterns indexed: 2).
Send a POST request to /pattern/evaluate:

```bash
curl -X POST http://localhost:8000/pattern/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "title": "User Authentication Service",
    "description": "A new microservice to handle user login using JWT tokens and Redis for session storage.",
    "user_stories": ["As a user, I want to login so I can access my account"]
  }'
```

Expected Result: The agent will search your ChromaDB for relevant company patterns (e.g., "Always use OAuth2", "Use PostgreSQL instead of Redis") and evaluate if your feature description violates or complies with them.

Use Case 3: Generate Non-Functional Requirements (NFRs) (Via API)
Scenario: You have a feature idea and need the AI to generate a list of NFRs (Security, Performance, etc.) for it.

```bash
curl -X POST http://localhost:8000/pattern/nfrs \
  -H "Content-Type: application/json" \
  -d '{
    "feature_description": "A real-time chat application for customer support.",
    "categories": ["Performance", "Security", "Scalability"]
  }'
```

Expected Result: The agent will return a detailed list of NFRs specific to those categories for a real-time chat app.

Use Case 4: Review a GitHub PR directly (Via CLI)
Scenario: You want to review a specific PR on GitHub without copying and pasting the diff.

Note: Requires GITHUB_TOKEN in your .env file.

Open a new terminal in the c:\dev\personal-agents directory.
Ensure your virtual environment is active: venv\Scripts\activate
Run the CLI tool:

```bash
github-review
```
Follow the interactive prompts:
Enter Owner: your-username
Enter Repo: your-repo
Enter PR Number: 1
Post as comment?: n
Expected Result: The agent uses the MCP GitHub server to fetch the PR, reviews it using Ollama, and prints the Markdown review directly in your terminal.

Use Case 5: Azure DevOps Interactive Session (Via CLI)
Scenario: You want to fetch features from your Azure DevOps project, evaluate them, and ingest your company wiki into the vector database.

Note: Requires AZURE_DEVOPS_ORG, AZURE_DEVOPS_PAT, and AZURE_DEVOPS_PROJECT in your .env file.

Open a new terminal in the c:\dev\personal-agents directory.
Ensure your virtual environment is active: venv\Scripts\activate
Run the CLI tool:

```bash
azure-devops
```
You will enter an interactive prompt. You can type:
wiki "Architecture Patterns": To ingest your Azure DevOps wiki pages into ChromaDB.
list: To see recent features in your ADO project.
eval <WorkItemID>: To evaluate a specific feature against the patterns you just ingested.
nfrs <WorkItemID>: To generate NFRs for that feature (it can even create new child NFR work items in ADO if configured to do so).
API Documentation (Swagger)
Since the server is running, you can open your browser and go to: http://127.0.0.1:8000/docs

This will show you a beautiful, interactive web interface where you can test all these endpoints directly by clicking "Try it out" and filling in the JSON fields.
## License

MIT
MIT
