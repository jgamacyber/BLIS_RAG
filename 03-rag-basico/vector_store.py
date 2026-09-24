"""
Armazenamento vetorial em memória, com persistência em disco.

Implementa busca exata (flat) por similaridade de cosseno / produto interno.
Não usa FAISS nem banco vetorial de propósito: para alguns milhares de chunks a
busca exata roda em milissegundos, e ver a conta explícita ensina mais do que
chamar uma biblioteca.

Contexto: o RAG original (Lewis et al., 2020) indexa 21M de passagens da
Wikipédia com FAISS + HNSW, uma aproximação (ANN) necessária nessa escala. Aqui
a escala é pequena, então a busca exata é a escolha certa — e serve de
referência de acurácia para comparar com um índice aproximado depois.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from chunking import Chunk

try:
    import numpy as np

    TEM_NUMPY = True
except ImportError:  # pragma: no cover - numpy está no requirements
    TEM_NUMPY = False


@dataclass
class Resultado:
    """Um chunk recuperado, com seu score e posição no ranking."""

    chunk: Chunk
    score: float
    rank: int

    def __repr__(self) -> str:  # pragma: no cover - only for debugging
        return f"<Resultado rank={self.rank} score={self.score:.4f} id={self.chunk.chunk_id}>"


# --------------------------------------------------------------------------- #
# Métricas de similaridade
# --------------------------------------------------------------------------- #


def produto_interno(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def norma(v: list[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def similaridade_cosseno(a: list[float], b: list[float]) -> float:
    """
    Cosseno = produto interno de vetores normalizados.

    Para vetores já normalizados (caso da maioria das APIs de embedding), cosseno
    e produto interno coincidem — foi por isso que o DPR pôde usar simplesmente o
    produto interno como função de similaridade.
    """
    na, nb = norma(a), norma(b)
    if na == 0 or nb == 0:
        return 0.0
    return produto_interno(a, b) / (na * nb)


# --------------------------------------------------------------------------- #
# Índice
# --------------------------------------------------------------------------- #


class VectorStore:
    """Índice plano de vetores com busca exata por cosseno."""

    def __init__(self, nome_modelo: str = "") -> None:
        self.chunks: list[Chunk] = []
        self.vetores: list[list[float]] = []
        self.nome_modelo = nome_modelo
        self._matriz = None  # cache numpy (n x d), normalizada por linha

    # ---------------------------- construção ---------------------------- #

    def adicionar(self, chunks: list[Chunk], vetores: list[list[float]]) -> None:
        if len(chunks) != len(vetores):
            raise ValueError(
                f"Quantidades diferentes: {len(chunks)} chunks e {len(vetores)} vetores"
            )
        self.chunks.extend(chunks)
        self.vetores.extend(vetores)
        self._matriz = None  # invalida o cache

    def _construir_matriz(self):
        """Pré-normaliza todos os vetores uma única vez (acelera a busca)."""
        if not TEM_NUMPY or not self.vetores:
            return None
        matriz = np.asarray(self.vetores, dtype=np.float32)
        normas = np.linalg.norm(matriz, axis=1, keepdims=True)
        normas[normas == 0] = 1.0
        return matriz / normas

    # ------------------------------ busca ------------------------------- #

    def buscar(self, vetor_consulta: list[float], top_k: int = 4) -> list[Resultado]:
        """Devolve os top_k chunks mais similares à consulta."""
        if not self.chunks:
            return []
        top_k = max(1, min(top_k, len(self.chunks)))

        if TEM_NUMPY:
            scores = self._scores_numpy(vetor_consulta)
        else:  # pragma: no cover - caminho de fallback
            scores = [similaridade_cosseno(vetor_consulta, v) for v in self.vetores]

        ordem = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            Resultado(chunk=self.chunks[i], score=float(scores[i]), rank=r + 1)
            for r, i in enumerate(ordem)
        ]

    def _scores_numpy(self, vetor_consulta: list[float]):
        if self._matriz is None:
            self._matriz = self._construir_matriz()
        consulta = np.asarray(vetor_consulta, dtype=np.float32)
        n = np.linalg.norm(consulta)
        if n > 0:
            consulta = consulta / n
        return self._matriz @ consulta

    def buscar_por_id(self, chunk_id: str) -> Chunk | None:
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk
        return None

    # --------------------------- persistência --------------------------- #

    def salvar(self, caminho: str | Path) -> None:
        caminho = Path(caminho)
        caminho.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "nome_modelo": self.nome_modelo,
            "dimensao": len(self.vetores[0]) if self.vetores else 0,
            "chunks": [asdict(c) for c in self.chunks],
            "vetores": self.vetores,
        }
        caminho.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def carregar(cls, caminho: str | Path) -> "VectorStore":
        caminho = Path(caminho)
        dados = json.loads(caminho.read_text(encoding="utf-8"))
        store = cls(nome_modelo=dados.get("nome_modelo", ""))
        store.chunks = [Chunk(**c) for c in dados["chunks"]]
        store.vetores = dados["vetores"]
        return store

    # ------------------------------ infos ------------------------------- #

    def __len__(self) -> int:
        return len(self.chunks)

    def resumo(self) -> str:
        docs = {c.doc_id for c in self.chunks}
        dim = len(self.vetores[0]) if self.vetores else 0
        return (
            f"{len(self.chunks)} chunks de {len(docs)} documentos | "
            f"dim={dim} | modelo={self.nome_modelo or 'n/d'}"
        )
