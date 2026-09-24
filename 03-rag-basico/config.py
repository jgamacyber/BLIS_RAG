"""
Configuração central do módulo 03 — RAG Básico.

Centraliza o acesso à API da OpenRouter (compatível com o SDK da OpenAI) e os
parâmetros do pipeline. Todos os valores podem ser sobrescritos pelo .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# A OpenRouter expõe endpoints compatíveis com a OpenAI, incluindo /embeddings.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
    """Parâmetros do pipeline de RAG."""

    # --- Credenciais e modelos ---
    api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    chat_model: str = field(
        default_factory=lambda: os.getenv("MODEL", "openai/gpt-4o-mini")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_MODEL", "openai/text-embedding-3-small"
        )
    )

    # "openrouter" usa a API real; "local" usa um embedder determinístico offline
    # (hashing), útil para testar o pipeline sem gastar créditos.
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openrouter").lower()
    )

    # --- Chunking ---
    # O paper original do RAG (Lewis et al., 2020) e o DPR (Karpukhin et al., 2020)
    # usam passagens disjuntas de ~100 palavras. Aqui o padrão é um pouco maior,
    # com sobreposição, que costuma funcionar melhor com LLMs modernos.
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "600")))
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100"))
    )

    # --- Recuperação e geração ---
    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K", "4")))
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0"))
    )
    max_tokens: int = field(default_factory=lambda: int(os.getenv("MAX_TOKENS", "600")))

    @property
    def usa_api(self) -> bool:
        """True se o pipeline deve chamar a API de embeddings da OpenRouter."""
        return self.embedding_provider == "openrouter"


SETTINGS = Settings()


def get_client(settings: Settings | None = None) -> OpenAI:
    """
    Devolve um cliente OpenAI apontando para a OpenRouter.

    Levanta um erro claro se a chave não estiver configurada, em vez de falhar
    com um 401 opaco no meio do pipeline.
    """
    settings = settings or SETTINGS
    if not settings.api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY não encontrada. Copie .env.example para .env e "
            "preencha a chave, ou rode com EMBEDDING_PROVIDER=local para o modo "
            "offline de demonstração."
        )
    return OpenAI(
        api_key=settings.api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/jgamacyber",
            "X-Title": "BLIS-RAG",
        },
    )


def resumo_config(settings: Settings | None = None) -> str:
    """Texto curto com a configuração ativa, usado nos banners do CLI."""
    s = settings or SETTINGS
    chave = "OK" if s.api_key else "AUSENTE"
    return (
        f"chat={s.chat_model} | embeddings={s.embedding_model} "
        f"({s.embedding_provider}) | chunk={s.chunk_size}/{s.chunk_overlap} "
        f"| top_k={s.top_k} | api_key={chave}"
    )
