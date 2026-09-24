"""
Configuração central do módulo 04 — RAG Avançado.

Mesmo núcleo do módulo 03, acrescido dos parâmetros das técnicas avançadas:
busca híbrida, HyDE, reescrita de consultas, reranking e reordenação de contexto.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class Settings:
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
    embedding_provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openrouter").lower()
    )

    # --- Chunking ---
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

    # ------------------------------------------------------------------ #
    # Parâmetros das técnicas avançadas
    # ------------------------------------------------------------------ #

    # Quantos candidatos a primeira etapa recupera antes do reranking.
    # O ganho do reranking vem justamente de recuperar mais e depois filtrar.
    candidatos: int = field(
        default_factory=lambda: int(os.getenv("CANDIDATOS", "20"))
    )

    # Busca híbrida: peso do denso na fusão ponderada (0..1). O RRF ignora
    # este valor, pois funde por posição e não por score.
    peso_denso: float = field(
        default_factory=lambda: float(os.getenv("PESO_DENSO", "0.5"))
    )
    # Constante do Reciprocal Rank Fusion (Cormack et al.): 60 é o padrão.
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))

    # HyDE: número de documentos hipotéticos amostrados (N na Eq. 8 do paper).
    hyde_n: int = field(default_factory=lambda: int(os.getenv("HYDE_N", "1")))
    # O paper usa 0.7 para gerar hipóteses diversas.
    hyde_temperatura: float = field(
        default_factory=lambda: float(os.getenv("HYDE_TEMPERATURA", "0.7"))
    )

    # Reescrita: quantas variações da consulta gerar no modo multi-query.
    n_reescritas: int = field(
        default_factory=lambda: int(os.getenv("N_REESCRITAS", "3"))
    )

    # RAPTOR: número de níveis da árvore acima das folhas.
    raptor_niveis: int = field(
        default_factory=lambda: int(os.getenv("RAPTOR_NIVEIS", "2"))
    )
    raptor_tamanho_cluster: int = field(
        default_factory=lambda: int(os.getenv("RAPTOR_TAMANHO_CLUSTER", "5"))
    )

    @property
    def usa_api(self) -> bool:
        return self.embedding_provider == "openrouter"


SETTINGS = Settings()


def get_client(settings: Settings | None = None) -> OpenAI:
    settings = settings or SETTINGS
    if not settings.api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY não encontrada. Copie .env.example para .env e "
            "preencha a chave, ou rode com EMBEDDING_PROVIDER=local para o modo "
            "offline (note que as técnicas que usam LLM ainda exigem a chave)."
        )
    return OpenAI(
        api_key=settings.api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers={
            "HTTP-Referer": "https://github.com/jgamacyber",
            "X-Title": "BLIS-RAG-Avancado",
        },
    )


def resumo_config(settings: Settings | None = None) -> str:
    s = settings or SETTINGS
    chave = "OK" if s.api_key else "AUSENTE"
    return (
        f"chat={s.chat_model} | embeddings={s.embedding_model} "
        f"({s.embedding_provider}) | top_k={s.top_k} | "
        f"candidatos={s.candidatos} | api_key={chave}"
    )
