"""
Avaliação do retriever.

Antes de culpar o LLM por uma resposta ruim, é preciso saber se a passagem certa
sequer foi recuperada. Estas métricas isolam a etapa de recuperação.

Métricas (padrão da literatura de IR, usadas em BEIR e no DPR):
- Hit@k / Recall@k: a passagem relevante aparece entre os k primeiros?
- MRR@k: 1/posição do primeiro acerto — premia colocar o certo no topo.
- Precision@k: fração dos k recuperados que são relevantes.
- nDCG@k: ganho acumulado com desconto logarítmico por posição; é a métrica
  principal do BEIR (Thakur et al., 2021).

O conjunto de avaliação fica em data/qa_eval.json, no formato:
    [{"pergunta": "...", "docs_relevantes": ["doc_id", ...]}, ...]
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from vector_store import Resultado

CAMINHO_EVAL = Path("data/qa_eval.json")


@dataclass
class ItemAvaliacao:
    pergunta: str
    docs_relevantes: list[str]
    resposta_referencia: str = ""


@dataclass
class Metricas:
    """Médias sobre todo o conjunto de avaliação."""

    k: int
    hit_rate: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    n_consultas: int = 0
    detalhes: list[dict] = field(default_factory=list)

    def imprimir(self, titulo: str = "") -> None:
        cabecalho = titulo or f"Métricas @{self.k}"
        print(f"\n{cabecalho}  ({self.n_consultas} consultas)")
        print("-" * 52)
        print(f"  Hit@{self.k}        {self.hit_rate:6.1%}   ao menos 1 relevante no top-{self.k}")
        print(f"  Recall@{self.k}     {self.recall:6.1%}   fração dos relevantes recuperados")
        print(f"  Precision@{self.k}  {self.precision:6.1%}   fração do top-{self.k} que é relevante")
        print(f"  MRR@{self.k}        {self.mrr:6.3f}   posição do primeiro acerto")
        print(f"  nDCG@{self.k}       {self.ndcg:6.3f}   qualidade do ranking")


# --------------------------------------------------------------------------- #
# Métricas por consulta
# --------------------------------------------------------------------------- #


def _docs_recuperados(resultados: list[Resultado]) -> list[str]:
    """Doc_ids na ordem do ranking, sem repetir (vários chunks do mesmo doc)."""
    vistos: list[str] = []
    for r in resultados:
        if r.chunk.doc_id not in vistos:
            vistos.append(r.chunk.doc_id)
    return vistos


def hit_e_mrr(recuperados: list[str], relevantes: set[str]) -> tuple[float, float]:
    for posicao, doc in enumerate(recuperados, start=1):
        if doc in relevantes:
            return 1.0, 1.0 / posicao
    return 0.0, 0.0


def recall_precision(
    recuperados: list[str], relevantes: set[str], k: int
) -> tuple[float, float]:
    topo = recuperados[:k]
    acertos = len(set(topo) & relevantes)
    recall = acertos / len(relevantes) if relevantes else 0.0
    precision = acertos / len(topo) if topo else 0.0
    return recall, precision


def ndcg(recuperados: list[str], relevantes: set[str], k: int) -> float:
    """
    nDCG com relevância binária.

    DCG = soma de rel_i / log2(i + 1); IDCG é o DCG do ranking perfeito.
    """
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, doc in enumerate(recuperados[:k], start=1)
        if doc in relevantes
    )
    ideal = min(len(relevantes), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------- #
# Avaliação do conjunto
# --------------------------------------------------------------------------- #


def carregar_conjunto(caminho: str | Path = CAMINHO_EVAL) -> list[ItemAvaliacao]:
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"Conjunto de avaliação não encontrado: {caminho}")
    dados = json.loads(caminho.read_text(encoding="utf-8"))
    return [
        ItemAvaliacao(
            pergunta=d["pergunta"],
            docs_relevantes=d["docs_relevantes"],
            resposta_referencia=d.get("resposta_referencia", ""),
        )
        for d in dados
    ]


def avaliar_retriever(
    recuperar_fn,
    conjunto: list[ItemAvaliacao],
    k: int = 4,
    verboso: bool = True,
) -> Metricas:
    """
    Avalia uma função de recuperação.

    `recuperar_fn(pergunta, k) -> list[Resultado]` — qualquer retriever que siga
    essa assinatura pode ser comparado (denso, BM25, híbrido, com reranking...).
    """
    m = Metricas(k=k, n_consultas=len(conjunto))
    if not conjunto:
        return m

    soma = {"hit": 0.0, "recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0}

    for item in conjunto:
        resultados = recuperar_fn(item.pergunta, k)
        recuperados = _docs_recuperados(resultados)
        relevantes = set(item.docs_relevantes)

        hit, mrr_q = hit_e_mrr(recuperados, relevantes)
        rec, prec = recall_precision(recuperados, relevantes, k)
        ndcg_q = ndcg(recuperados, relevantes, k)

        soma["hit"] += hit
        soma["recall"] += rec
        soma["precision"] += prec
        soma["mrr"] += mrr_q
        soma["ndcg"] += ndcg_q

        m.detalhes.append(
            {
                "pergunta": item.pergunta,
                "relevantes": sorted(relevantes),
                "recuperados": recuperados[:k],
                "hit": bool(hit),
                "mrr": round(mrr_q, 3),
                "ndcg": round(ndcg_q, 3),
            }
        )

        if verboso:
            marca = "OK " if hit else "ERRO"
            print(f"  [{marca}] {item.pergunta[:58]:<58} mrr={mrr_q:.2f}")

    n = len(conjunto)
    m.hit_rate = soma["hit"] / n
    m.recall = soma["recall"] / n
    m.precision = soma["precision"] / n
    m.mrr = soma["mrr"] / n
    m.ndcg = soma["ndcg"] / n
    return m


def comparar(metricas: dict[str, Metricas]) -> None:
    """Tabela comparativa entre configurações (ex.: chunking A vs B)."""
    if not metricas:
        return
    k = next(iter(metricas.values())).k
    print(f"\nComparação @{k}")
    print("-" * 74)
    print(f"{'Configuração':<28} {'Hit':>8} {'Recall':>8} {'MRR':>8} {'nDCG':>8}")
    print("-" * 74)
    for nome, m in metricas.items():
        print(
            f"{nome:<28} {m.hit_rate:>7.1%} {m.recall:>8.1%} "
            f"{m.mrr:>8.3f} {m.ndcg:>8.3f}"
        )
    print("-" * 74)
