from pathlib import Path
import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Explicit OpenAI (optional)
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""
    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = ""
    gemini_api_key: str = ""
    # Generic overrides (highest priority)
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    router_model: str = ""
    llm_fallback_models: str = ""
    embed_model: str = "BAAI/bge-small-en-v1.5"
    embed_dim: int = 384
    tavily_api_key: str = ""
    app_user: str = ""
    app_password: str = ""
    data_dir: str = "./data"
    chunk_tokens: int = 800
    chunk_overlap: float = 0.15
    top_k: int = 8
    max_hops: int = 2

    def _is_placeholder(self, key: str) -> bool:
        k = (key or "").strip()
        return (not k) or k in {"sk-...", "sk-xxx"} or k.endswith("...")

    @property
    def effective_base_url(self) -> str:
        if self.llm_base_url:
            return self.llm_base_url
        if self.openai_base_url:
            return self.openai_base_url
        if not self._is_placeholder(self.openai_api_key):
            return "https://api.openai.com/v1"
        ork = self.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if ork:
            return "https://openrouter.ai/api/v1"
        gk = self.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if gk:
            return "https://generativelanguage.googleapis.com/v1beta/openai/"
        return "https://api.openai.com/v1"

    @property
    def effective_api_key(self) -> str:
        if self.llm_api_key:
            return self.llm_api_key
        if not self._is_placeholder(self.openai_api_key):
            return self.openai_api_key
        ork = self.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if ork:
            return ork
        gk = self.gemini_api_key or os.environ.get("GEMINI_API_KEY", "")
        if gk:
            return gk
        return self.openai_api_key

    @property
    def _using_gemini(self) -> bool:
        return "generativelanguage.googleapis.com" in self.effective_base_url

    @property
    def effective_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        if self._using_gemini:
            return "gemini-2.0-flash"
        base = self.openrouter_model or self.openai_model or "gpt-4o-mini"
        using_or = "openrouter.ai" in self.effective_base_url
        if using_or and "/" not in base:
            return "openai/" + base
        return base

    @property
    def effective_router_model(self) -> str:
        if self.router_model:
            return self.router_model
        return self.effective_model

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def lancedb_path(self) -> Path:
        p = self.data_path / "lancedb"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def uploads_path(self) -> Path:
        p = self.data_path / "uploads"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def app_db(self) -> str:
        return str(self.data_path / "ragbuddy.db")

    @property
    def graph_db(self) -> str:
        return str(self.data_path / "graph.db")


settings = Settings()


def ensure_dirs() -> None:
    settings.data_path
    settings.lancedb_path
    settings.uploads_path
