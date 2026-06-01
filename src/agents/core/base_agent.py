"""Base agent class with common functionality for all agents."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from rich.console import Console

from agents.core.llm_client import OllamaClient, ChatResponse
from agents.core.config import get_settings


class AgentContext(BaseModel):
    """Context passed to agent during execution."""

    user_input: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    history: list[dict[str, str]] = Field(default_factory=list)


class AgentResult(BaseModel):
    """Result returned by agent after execution."""

    success: bool
    output: str
    reasoning: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    suggestions: list[str] = Field(default_factory=list)


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Provides common functionality:
    - LLM client management
    - Conversation history
    - System prompt handling
    - Logging and output
    """

    def __init__(
        self,
        name: str,
        description: str,
        system_prompt: Optional[str] = None,
        llm_client: Optional[OllamaClient] = None,
    ):
        """Initialize the base agent.

        Args:
            name: Agent name for identification.
            description: Brief description of agent's purpose.
            system_prompt: System prompt to guide agent behavior.
            llm_client: Optional pre-configured LLM client.
        """
        self.name = name
        self.description = description
        self._system_prompt = system_prompt or self._default_system_prompt()
        self._client = llm_client or OllamaClient()
        self._history: list[BaseMessage] = []
        self._console = Console()
        self._settings = get_settings()

    @property
    def llm(self) -> OllamaClient:
        """Get the LLM client."""
        return self._client

    @property
    def system_prompt(self) -> str:
        """Get the system prompt."""
        return self._system_prompt

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        """Set a new system prompt."""
        self._system_prompt = value

    @property
    def history(self) -> list[BaseMessage]:
        """Get conversation history."""
        return self._history.copy()

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to conversation history.

        Args:
            role: Either 'user' or 'assistant'.
            content: Message content.
        """
        if role == "user":
            self._history.append(HumanMessage(content=content))
        elif role == "assistant":
            self._history.append(AIMessage(content=content))

    @abstractmethod
    def _default_system_prompt(self) -> str:
        """Return the default system prompt for this agent type.

        Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def execute(self, context: AgentContext) -> AgentResult:
        """Execute the agent's main task.

        Args:
            context: Execution context with user input and metadata.

        Returns:
            AgentResult with the execution outcome.
        """
        pass

    def chat(self, message: str, remember: bool = True) -> ChatResponse:
        """Send a chat message to the LLM.

        Args:
            message: User message.
            remember: Whether to add to conversation history.

        Returns:
            ChatResponse from the LLM.
        """
        response = self._client.chat(
            message=message,
            system_prompt=self._system_prompt,
            history=self._history if remember else None,
        )

        if remember:
            self.add_to_history("user", message)
            self.add_to_history("assistant", response.content)

        return response

    async def achat(self, message: str, remember: bool = True) -> ChatResponse:
        """Async version of chat.

        Args:
            message: User message.
            remember: Whether to add to conversation history.

        Returns:
            ChatResponse from the LLM.
        """
        response = await self._client.achat(
            message=message,
            system_prompt=self._system_prompt,
            history=self._history if remember else None,
        )

        if remember:
            self.add_to_history("user", message)
            self.add_to_history("assistant", response.content)

        return response

    def log(self, message: str, style: str = "default") -> None:
        """Log a message to console with styling.

        Args:
            message: Message to log.
            style: Rich style (e.g., 'bold', 'green', 'red').
        """
        prefix = f"[{self.name}]"
        if style == "default":
            self._console.print(f"{prefix} {message}")
        else:
            self._console.print(f"{prefix} {message}", style=style)

    def log_info(self, message: str) -> None:
        """Log an info message."""
        self.log(message, style="blue")

    def log_success(self, message: str) -> None:
        """Log a success message."""
        self.log(message, style="green")

    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        self.log(message, style="yellow")

    def log_error(self, message: str) -> None:
        """Log an error message."""
        self.log(message, style="red bold")

    def is_ready(self) -> bool:
        """Check if the agent is ready to execute.

        Returns:
            True if LLM is available and agent is configured.
        """
        return self._client.is_available()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model='{self._client.model_name}')"
