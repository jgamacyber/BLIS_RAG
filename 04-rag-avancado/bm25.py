"""
BM25 (Okapi) implementado do zero, sem dependências.

O BM25 é o baseline lexical de referência em recuperação de informação. O
benchmark BEIR (Thakur et al., 2021) mostrou que ele permanece robusto em
cenário zero-shot, superando vários modelos densos fora da distribuição de
treino — motivo pelo qual ele entra na busca híbrida em vez de ser descartado.

Fórmula:

    score(D, Q) = Σ  IDF(q) · ────────── f(q,D) · (k1 + 1) ──────────
                  q∈Q          f(q,D) + k1 · (1 - b + b · |D|/avgdl)

    IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )

Onde:
    f(q,D)  frequência do termo q no documento D
    |D|     comprimento do documento em tokens
    avgdl   comprimento médio dos documentos do corpus
    N       número de documentos
    n(q)    número de documentos que contêm q
    k1      saturação da frequência do termo (padrão 1.5)
    b       normalização por comprimento (padrão 0.75)

O "+1" dentro do log é a variante que garante IDF não negativo, evitando que
termos presentes em mais da metade do corpus recebam peso negativo.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from chunking import Chunk
from vector_store import Resultado

# Stopwords do português: termos frequentes demais para discriminar documentos.
STOPWORDS_PT = {
    "a", "à", "às", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles",
    "aquilo", "as", "até", "com", "como", "da", "das", "de", "dela", "delas",
    "dele", "deles", "depois", "do", "dos", "e", "é", "ela", "elas", "ele",
    "eles", "em", "entre", "era", "eram", "essa", "essas", "esse", "esses",
    "esta", "está", "estas", "este", "estes", "eu", "foi", "foram", "há",
    "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu",
    "meus", "minha", "minhas", "muito", "na", "não", "nas", "nem", "no",
    "nos", "nós", "nossa", "nossas", "nosso", "nossos", "num", "numa", "o",
    "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual",
    "quando", "que", "quem", "se", "sem", "ser", "seu", "seus", "só", "sua",
    "suas", "também", "te", "tem", "têm", "teu", "teus", "tu", "tua", "tuas",
    "um", "uma", "umas", "uns", "você", "vocês",
}

_TOKEN = re.compile(r"[a-z0-9]+")


def remover_acentos(texto: str) -> str:
    """Normaliza acentuação: 'função' e 'funcao' viram o mesmo token."""
    nfkd = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenizar(texto: str, remover_stopwords: bool = True) -> list[str]:
    """Minúsculas, sem acentos, apenas alfanuméricos, sem stopwords."""
    tokens = _TOKEN.findall(remover_acentos(texto.lower()))
    if remover_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_PT and len(t) > 1]
    return tokens


class BM25:
    """Índice BM25 sobre uma lista de chunks."""

    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
        remover_stopwords: bool = True,
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.remover_stopwords = remover_stopwords

        # Cada chunk é indexado com título + texto: o título costuma conter
        # termos discriminativos (é a mesma intuição do [SEP] título do DPR).
        self.documentos: list[list[str]] = [
            tokenizar(f"{c.titulo} {c.texto}", remover_stopwords) for c in chunks
        ]
        self.frequencias: list[Counter] = [Counter(d) for d in self.documentos]
        self.comprimentos: list[int] = [len(d) for d in self.documentos]

        self.n_docs = len(self.documentos)
        self.avgdl = (
            sum(self.comprimentos) / self.n_docs if self.n_docs else 0.0
        )
        self.idf: dict[str, float] = self._calcular_idf()

    def _calcular_idf(self) -> dict[str, float]:
        """Conta em quantos documentos cada termo aparece e deriva o IDF."""
        df: Counter = Counter()
        for freq in self.frequencias:
            df.update(freq.keys())

        return {
            termo: math.log((self.n_docs - n + 0.5) / (n + 0.5) + 1.0)
            for termo, n in df.items()
        }

    def score(self, consulta_tokens: list[str], indice_doc: int) -> float:
        """Score BM25 de um documento para uma consulta já tokenizada."""
        freq = self.frequencias[indice_doc]
        comprimento = self.comprimentos[indice_doc]
        if comprimento == 0:
            return 0.0

        # Normalização por comprimento, constante dentro do documento.
        norma = self.k1 * (1 - self.b + self.b * comprimento / self.avgdl)

        total = 0.0
        for termo in consulta_tokens:
            f = freq.get(termo, 0)
            if f == 0:
                continue
            total += self.idf.get(termo, 0.0) * (f * (self.k1 + 1)) / (f + norma)
        return total

    def buscar(self, consulta: str, top_k: int = 10) -> list[Resultado]:
        """Ranqueia os chunks pela pontuação BM25."""
        if not self.chunks:
            return []

        tokens = tokenizar(consulta, self.remover_stopwords)
        if not tokens:
            return []

        scores = [(i, self.score(tokens, i)) for i in range(self.n_docs)]
        scores = [(i, s) for i, s in scores if s > 0]
        scores.sort(key=lambda par: par[1], reverse=True)

        return [
            Resultado(chunk=self.chunks[i], score=float(s), rank=r + 1)
            for r, (i, s) in enumerate(scores[:top_k])
        ]

    def termos_da_consulta(self, consulta: str) -> list[tuple[str, float]]:
        """
        Termos da consulta com seus IDFs, do mais raro ao mais comum.

        Útil para depurar: mostra quais palavras realmente pesam na busca
        lexical e quais foram descartadas por serem stopwords ou não existirem
        no corpus.
        """
        tokens = tokenizar(consulta, self.remover_stopwords)
        pares = [(t, self.idf.get(t, 0.0)) for t in dict.fromkeys(tokens)]
        return sorted(pares, key=lambda p: p[1], reverse=True)

    def __len__(self) -> int:
        return self.n_docs

    def resumo(self) -> str:
        return (
            f"BM25: {self.n_docs} documentos | vocabulário={len(self.idf)} termos "
            f"| avgdl={self.avgdl:.1f} tokens | k1={self.k1} b={self.b}"
        )
