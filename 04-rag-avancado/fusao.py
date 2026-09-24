"""
Fusão de rankings para busca híbrida.

O problema: o BM25 produz scores não normalizados que podem ir de 0 a valores
arbitrariamente altos; a similaridade de cosseno vive entre -1 e 1. Somar os
dois diretamente é incoerente.

Duas soluções implementadas:

1. Reciprocal Rank Fusion (RRF) — Cormack, Clarke & Buettcher (2009).
   Ignora os scores e usa apenas as posições:

       score_RRF(d) = Σ  1 / (k + posição(d, ranking_i))
                      i

   Com k = 60 (padrão), o 1º lugar contribui 1/61 e o 10º contribui 1/70.
   Não exige normalização nem calibração e é robusto a escalas incompatíveis.

2. Soma ponderada com normalização min-max. Preserva a magnitude relativa dos
   scores, mas é sensível a outliers e à distribuição de cada consulta.

Na prática o RRF costuma ser a escolha mais segura; a soma ponderada só ganha
quando se sabe que um recuperador é consistentemente melhor que o outro.
"""

from __future__ import annotations

from vector_store import Resultado


def _chave(resultado: Resultado) -> str:
    return resultado.chunk.chunk_id


def reciprocal_rank_fusion(
    rankings: list[list[Resultado]],
    k: int = 60,
    top_k: int = 10,
    pesos: list[float] | None = None,
) -> list[Resultado]:
    """
    Funde vários rankings pela soma dos recíprocos das posições.

    `pesos` permite dar mais influência a um recuperador; por padrão todos
    valem o mesmo, que é o comportamento clássico do RRF.
    """
    rankings = [r for r in rankings if r]
    if not rankings:
        return []

    if pesos is None:
        pesos = [1.0] * len(rankings)
    if len(pesos) != len(rankings):
        raise ValueError("pesos e rankings devem ter o mesmo comprimento")

    scores: dict[str, float] = {}
    chunks: dict[str, Resultado] = {}
    procedencia: dict[str, list[int]] = {}

    for idx, (ranking, peso) in enumerate(zip(rankings, pesos)):
        for resultado in ranking:
            chave = _chave(resultado)
            # resultado.rank é 1-based, como o RRF espera.
            scores[chave] = scores.get(chave, 0.0) + peso / (k + resultado.rank)
            chunks.setdefault(chave, resultado)
            procedencia.setdefault(chave, []).append(idx)

    ordenados = sorted(scores.items(), key=lambda par: par[1], reverse=True)

    finais: list[Resultado] = []
    for posicao, (chave, score) in enumerate(ordenados[:top_k], start=1):
        base = chunks[chave]
        finais.append(
            Resultado(
                chunk=base.chunk,
                score=float(score),
                rank=posicao,
            )
        )
    return finais


def _normalizar_minmax(resultados: list[Resultado]) -> dict[str, float]:
    """Mapeia os scores de um ranking para o intervalo [0, 1]."""
    if not resultados:
        return {}
    scores = [r.score for r in resultados]
    menor, maior = min(scores), max(scores)
    intervalo = maior - menor

    if intervalo == 0:
        # Todos iguais: atribui 1.0 a todos em vez de dividir por zero.
        return {_chave(r): 1.0 for r in resultados}
    return {_chave(r): (r.score - menor) / intervalo for r in resultados}


def soma_ponderada(
    denso: list[Resultado],
    lexical: list[Resultado],
    peso_denso: float = 0.5,
    top_k: int = 10,
) -> list[Resultado]:
    """
    Combina dois rankings por soma ponderada dos scores normalizados.

    Documentos ausentes de um dos rankings recebem 0 naquele componente, o que
    penaliza (corretamente) quem só foi encontrado por um dos recuperadores.
    """
    if not 0.0 <= peso_denso <= 1.0:
        raise ValueError("peso_denso deve estar entre 0 e 1")

    norm_denso = _normalizar_minmax(denso)
    norm_lexical = _normalizar_minmax(lexical)

    chunks: dict[str, Resultado] = {}
    for resultado in list(denso) + list(lexical):
        chunks.setdefault(_chave(resultado), resultado)

    combinados: dict[str, float] = {}
    for chave in chunks:
        combinados[chave] = (
            peso_denso * norm_denso.get(chave, 0.0)
            + (1.0 - peso_denso) * norm_lexical.get(chave, 0.0)
        )

    ordenados = sorted(combinados.items(), key=lambda par: par[1], reverse=True)
    return [
        Resultado(chunk=chunks[chave].chunk, score=float(score), rank=posicao)
        for posicao, (chave, score) in enumerate(ordenados[:top_k], start=1)
    ]


def deduplicar(resultados: list[Resultado]) -> list[Resultado]:
    """Remove chunks repetidos preservando a ordem e renumerando os ranks."""
    vistos: set[str] = set()
    finais: list[Resultado] = []
    for resultado in resultados:
        chave = _chave(resultado)
        if chave in vistos:
            continue
        vistos.add(chave)
        finais.append(
            Resultado(chunk=resultado.chunk, score=resultado.score, rank=len(finais) + 1)
        )
    return finais
