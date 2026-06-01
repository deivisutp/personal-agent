"""Ollama LLM client wrapper with LangChain integration."""

from typing import Any, Optional

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from agents.core.config import get_settings, OllamaSettings


class ChatResponse(BaseModel):
    """Structured response from the LLM."""

    content: str
    model: str
    tokens_used: Optional[int] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OllamaClient:
    """Wrapper for Ollama LLM with LangChain integration."""

    def __init__(
        self,
        settings: Optional[OllamaSettings] = None,
        model: Optional[str] = None,
    ):
        """Initialize the Ollama client.

        Args:
            settings: Ollama settings. If None, loads from environment.
            model: Override model name. If None, uses settings.
        """
        self._settings = settings or get_settings().ollama
        self._model = model or self._settings.model

        self._llm = ChatOllama(
            base_url=self._settings.base_url,
            model=self._model,
            temperature=self._settings.temperature,
            num_predict=self._settings.max_tokens,
        )

        self._embeddings: Optional[OllamaEmbeddings] = None

    @property
    def llm(self) -> ChatOllama:
        """Get the underlying LangChain LLM instance."""
        return self._llm

    @property
    def embeddings(self) -> OllamaEmbeddings:
        """Get or create the embeddings model (lazy initialization)."""
        if self._embeddings is None:
            self._embeddings = OllamaEmbeddings(
                base_url=self._settings.base_url,
                model=self._settings.embedding_model,
            )
        return self._embeddings

    @property
    def model_name(self) -> str:
        """Get the current model name."""
        return self._model

    def chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[BaseMessage]] = None,
    ) -> ChatResponse:
        """Send a chat message and get a response.

        Args:
            message: The user message.
            system_prompt: Optional system prompt to set context.
            history: Optional conversation history.

        Returns:
            ChatResponse with the LLM's response.
        """
        messages: list[BaseMessage] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if history:
            messages.extend(history)

        messages.append(HumanMessage(content=message))

        response = self._llm.invoke(messages)

        return ChatResponse(
            content=response.content,
            model=self._model,
            metadata=response.response_metadata if hasattr(response, "response_metadata") else {},
        )

    async def achat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[BaseMessage]] = None,
    ) -> ChatResponse:
        """Async version of chat.

        Args:
            message: The user message.
            system_prompt: Optional system prompt to set context.
            history: Optional conversation history.

        Returns:
            ChatResponse with the LLM's response.
        """
        messages: list[BaseMessage] = []

        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        if history:
            messages.extend(history)

        messages.append(HumanMessage(content=message))

        response = await self._llm.ainvoke(messages)

        return ChatResponse(
            content=response.content,
            model=self._model,
            metadata=response.response_metadata if hasattr(response, "response_metadata") else {},
        )

    async def astream_chat(
        self,
        message: str,
        system_prompt: Optional[str] = None,
        history: Optional[list[BaseMessage]] = None,
    ) -> AsyncIterator[str]:
        """Stream a chat response token-by-token (yields content deltas).

        Args:
            message: The user message.
            system_prompt: Optional system prompt.
            history: Optional conversation history.

        Yields:
            String content deltas from the model.
        """
        messages: list[BaseMessage] = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        if history:
            messages.extend(history)
        messages.append(HumanMessage(content=message))

        async for chunk in self._llm.astream(messages):
            content = getattr(chunk, "content", None)
            if content:
                yield content

    def invoke_with_template(
        self,
        template: ChatPromptTemplate,
        **kwargs: Any,
    ) -> ChatResponse:
        """Invoke the LLM with a prompt template.

        Args:
            template: LangChain prompt template.
            **kwargs: Variables to fill in the template.

        Returns:
            ChatResponse with the LLM's response.
        """
        chain = template | self._llm
        response = chain.invoke(kwargs)

        return ChatResponse(
            content=response.content,
            model=self._model,
            metadata=response.response_metadata if hasattr(response, "response_metadata") else {},
        )

    def embed_text(self, text: str) -> list[float]:
        """Generate embeddings for a single text.

        Args:
            text: Text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        return self.embeddings.embed_query(text)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return self.embeddings.embed_documents(texts)

    def is_available(self) -> bool:
        """Check if Ollama server is available and model is loaded."""
        try:
            import httpx

            response = httpx.get(f"{self._settings.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return any(m.get("name", "").startswith(self._model) for m in models)
            return False
        except Exception:
            return False
