# Personal Agents — Architecture

```mermaid
graph TB
    %% ── User Layer ──
    User["👤 User"]

    subgraph CLI["🖥️ Console Interface (CLI)"]
        direction TB
        PRCmd["pr-agent<br/><i>Interactive PR review</i>"]
        PatternCmd["pattern-agent<br/><i>ingest / evaluate / nfrs</i>"]
        AzureCmd["azure-devops<br/><i>list / eval / nfrs / wiki</i>"]
        GitHubCmd["github<br/><i>owner/repo#number</i>"]
    end

    subgraph API["🌐 FastAPI REST API"]
        direction TB
        PRRouter["/pr-review/*"]
        PatternRouter["/pattern/*"]
    end

    User --> CLI
    User --> API

    %% ── Agent Layer ──
    subgraph Agents["🤖 Agent Layer"]
        direction TB
        PRAgent["PR Review Agent<br/><i>Code review, security,<br/>performance analysis</i>"]
        PatternAgent["Pattern & NFR Agent<br/><i>Feature evaluation,<br/>NFR generation,<br/>pattern compliance</i>"]
    end

    PRCmd --> PRAgent
    GitHubCmd --> PRAgent
    PRRouter --> PRAgent

    PatternCmd --> PatternAgent
    AzureCmd --> PatternAgent
    PatternRouter --> PatternAgent

    %% ── Core Layer ──
    subgraph Core["⚙️ Core"]
        direction LR
        BaseAgent["BaseAgent<br/><i>History, prompt handling,<br/>logging</i>"]
        LLMClient["OllamaClient<br/><i>LangChain ChatOllama +<br/>OllamaEmbeddings</i>"]
    end

    PRAgent --> BaseAgent
    PatternAgent --> BaseAgent
    BaseAgent --> LLMClient

    %% ── LLM ──
    subgraph Ollama["🧠 Ollama (Local LLM)"]
        direction LR
        Mistral["mistral-nemo"]
        Qwen["qwen2.5:7b"]
    end

    LLMClient -- "chat / embeddings" --> Ollama

    %% ── RAG Pipeline ──
    subgraph RAG["📚 RAG Pipeline"]
        direction TB
        DocLoader["DocumentLoader<br/><i>Chunking<br/>(1000 chars, 200 overlap)</i>"]
        VectorStore["ChromaVectorStore<br/><i>Similarity search</i>"]
        ChromaDB[("ChromaDB<br/><i>Persistent vector storage</i>")]
    end

    PatternAgent -- "ingest patterns<br/>(files, wiki, text)" --> DocLoader
    DocLoader -- "DocumentChunks" --> VectorStore
    VectorStore --> ChromaDB
    PatternAgent -- "retrieve relevant<br/>patterns (top-k)" --> VectorStore

    %% ── MCP Integrations ──
    subgraph MCP["🔌 MCP Clients (stdio)"]
        direction TB
        AzureMCP["AzureDevOpsMCPClient<br/><i>Work items, wiki,<br/>user stories</i>"]
        GitHubMCP["GitHubMCPClient<br/><i>PRs, diffs, comments</i>"]
    end

    PatternAgent -- "fetch features,<br/>stories, wiki" --> AzureMCP
    PRAgent -- "fetch PRs,<br/>post comments" --> GitHubMCP

    %% ── External Services ──
    AzureDevOps["☁️ Azure DevOps<br/><i>Work Items / Wiki</i>"]
    GitHub["🐙 GitHub<br/><i>Repositories / PRs</i>"]

    AzureMCP -- "MCP over stdio" --> AzureDevOps
    GitHubMCP -- "MCP over stdio" --> GitHub

    %% ── Styling ──
    classDef user fill:#4A90D9,stroke:#2C5F8A,color:#fff,font-weight:bold
    classDef cli fill:#2D3748,stroke:#4A5568,color:#E2E8F0
    classDef api fill:#2D3748,stroke:#4A5568,color:#E2E8F0
    classDef agent fill:#6B46C1,stroke:#553C9A,color:#fff,font-weight:bold
    classDef core fill:#2B6CB0,stroke:#2C5282,color:#fff
    classDef llm fill:#D69E2E,stroke:#B7791F,color:#1A202C,font-weight:bold
    classDef rag fill:#2F855A,stroke:#276749,color:#fff
    classDef db fill:#38A169,stroke:#2F855A,color:#fff,font-weight:bold
    classDef mcp fill:#DD6B20,stroke:#C05621,color:#fff
    classDef ext fill:#718096,stroke:#4A5568,color:#fff

    class User user
    class PRCmd,PatternCmd,AzureCmd,GitHubCmd cli
    class PRRouter,PatternRouter api
    class PRAgent,PatternAgent agent
    class BaseAgent,LLMClient core
    class Mistral,Qwen llm
    class DocLoader,VectorStore rag
    class ChromaDB db
    class AzureMCP,GitHubMCP mcp
    class AzureDevOps,GitHub ext
```

## Flow Summary

| Flow | Path |
|------|------|
| **PR Review** | CLI / API → PR Review Agent → Ollama (mistral-nemo / qwen2.5) |
| **GitHub PR Review** | CLI / API → PR Review Agent → GitHub MCP → GitHub API → Ollama |
| **Feature Evaluation** | CLI / API → Pattern Agent → Azure DevOps MCP → RAG (ChromaDB) → Ollama |
| **NFR Generation** | CLI / API → Pattern Agent → Azure DevOps MCP → RAG (ChromaDB) → Ollama |
| **Pattern Ingestion** | CLI / API → DocumentLoader (chunking) → ChromaDB |
| **Wiki Ingestion** | CLI → Pattern Agent → Azure DevOps MCP → DocumentLoader → ChromaDB |
