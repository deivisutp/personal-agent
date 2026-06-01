"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Each nested BaseSettings is instantiated independently (via `default_factory`),
# so each one needs its own env_file config to actually read `.env`. Without
# this, only OS environment variables are picked up and `.env` is ignored for
# nested groups (Azure DevOps, Ollama, etc.).
_ENV_FILE_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
)


class OllamaSettings(BaseSettings):
    """Ollama-specific configuration."""

    model_config = _ENV_FILE_CONFIG

    base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    model: str = Field(default="mistral-nemo", alias="OLLAMA_MODEL")
    embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")
    timeout: int = Field(default=120, description="Request timeout in seconds")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=None, description="Max tokens for response")


class ChromaSettings(BaseSettings):
    """ChromaDB configuration."""

    model_config = _ENV_FILE_CONFIG

    persist_dir: Path = Field(default=Path("./data/chroma"), alias="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="patterns", description="Default collection name")


class GitHubSettings(BaseSettings):
    """GitHub MCP configuration."""

    model_config = _ENV_FILE_CONFIG

    token: Optional[str] = Field(default=None, alias="GITHUB_TOKEN")
    mcp_server: str = Field(
        default="npx -y @modelcontextprotocol/server-github",
        alias="GITHUB_MCP_SERVER",
    )


class AzureDevOpsSettings(BaseSettings):
    """Azure DevOps MCP configuration."""

    model_config = _ENV_FILE_CONFIG

    organization: Optional[str] = Field(default=None, alias="AZURE_DEVOPS_ORG")
    pat: Optional[str] = Field(default=None, alias="AZURE_DEVOPS_PAT")
    project: Optional[str] = Field(default=None, alias="AZURE_DEVOPS_PROJECT")


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    github: GitHubSettings = Field(default_factory=GitHubSettings)
    azure_devops: AzureDevOpsSettings = Field(default_factory=AzureDevOpsSettings)

    @classmethod
    def load(cls) -> "Settings":
        """Load settings from environment and .env file."""
        return cls(
            ollama=OllamaSettings(),
            chroma=ChromaSettings(),
            github=GitHubSettings(),
            azure_devops=AzureDevOpsSettings(),
        )


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.load()
    return _settings
