"""
Reordenação do contexto contra o efeito "Lost in the Middle".

Liu et al. (2023) mostraram que o desempenho dos LLMs em contextos longos segue
uma curva em U: é alto quando a informação relevante está no começo (viés de
primazia) ou no fim (viés de recência) do contexto, e cai significativamente
quando ela está no meio.

O achado mais duro do paper: com 20 documentos, colocar a passagem correta no
meio do contexto deixou o GPT-3.5-Turbo ABAIXO do desempenho closed-book (56,1%),
ou seja, pior do que não fornecer documento nenhum.

A mitigação é barata e não exige chamada extra ao LLM: depois de ranquear as
passagens por relevância, reordená-las fisicamente de modo que as mais
relevantes ocupem as pontas do prompt e as menos relevantes fiquem no meio.

    ranking:   1  2  3  4  5  6  7
    prompt:    1  3  5  7  6  4  2
               └── começo ──┘└─ fim ─┘

O segundo achado com efeito prático: "recuperar mais não é recuperar melhor".
O desempenho satura muito antes da revocação do recuperador — passar de 20 para
50 documentos melhorou só ~1,5% no GPT-3.5. Daí a função `truncar_contexto`.
"""

from __future__ import annotations

from vector_store import Resultado


def ordenar_bordas(resultados: list[Resultado]) -> list[Resultado]:
    """
    Coloca as passagens mais relevantes nas extremidades do contexto.

    A de rank 1 vai para a primeira posição, a de rank 2 para a última, a de
    rank 3 para a segunda, e assim por diante — o pior colocado acaba no meio,
    que é a zona morta da curva em U.
    """
    if len(resultados) <= 2:
        return list(resultados)

    inicio: list[Resultado] = []
    fim: list[Resultado] = []

    for i, resultado in enumerate(resultados):
        (inicio if i % 2 == 0 else fim).append(resultado)

    ordenados = inicio + list(reversed(fim))
    return _renumerar(ordenados)


def ordenar_relevancia_decrescente(resultados: list[Resultado]) -> list[Resultado]:
    """Ordem natural do ranking: mais relevante primeiro. Explora só a primazia."""
    return _renumerar(sorted(resultados, key=lambda r: r.score, reverse=True))


def ordenar_relevancia_crescente(resultados: list[Resultado]) -> list[Resultado]:
    """
    Menos relevante primeiro, mais relevante por último.

    Explora o viés de recência: a informação mais importante fica imediatamente
    antes da pergunta no prompt. Em prompts curtos costuma funcionar tão bem
    quanto a ordenação por bordas.
    """
    return _renumerar(sorted(resultados, key=lambda r: r.score))


def ordenar_por_documento(resultados: list[Resultado]) -> list[Resultado]:
    """
    Agrupa passagens do mesmo documento e as coloca na ordem original do texto.

    Útil quando o corpus tem narrativa ou encadeamento lógico: passagens
    consecutivas do mesmo documento fora de ordem confundem o gerador. Os
    documentos são ordenados pelo melhor score de suas passagens.
    """
    grupos: dict[str, list[Resultado]] = {}
    for resultado in resultados:
        grupos.setdefault(resultado.chunk.doc_id, []).append(resultado)

    ordem_docs = sorted(
        grupos.items(),
        key=lambda item: max(r.score for r in item[1]),
        reverse=True,
    )

    ordenados: list[Resultado] = []
    for _, passagens in ordem_docs:
        ordenados.extend(sorted(passagens, key=lambda r: r.chunk.posicao))
    return _renumerar(ordenados)


def _renumerar(resultados: list[Resultado]) -> list[Resultado]:
    """Reatribui o campo rank conforme a nova ordem física no prompt."""
    return [
        Resultado(chunk=r.chunk, score=r.score, rank=i)
        for i, r in enumerate(resultados, start=1)
    ]


def truncar_contexto(
    resultados: list[Resultado], max_chars: int = 8000
) -> list[Resultado]:
    """
    Corta o contexto por orçamento de caracteres, preservando os mais relevantes.

    Aplicar ANTES da reordenação: primeiro decide-se o que entra, depois onde
    cada passagem fica.
    """
    selecionados: list[Resultado] = []
    total = 0
    for resultado in sorted(resultados, key=lambda r: r.score, reverse=True):
        tamanho = len(resultado.chunk.texto)
        if total + tamanho > max_chars and selecionados:
            break
        selecionados.append(resultado)
        total += tamanho
    return _renumerar(sorted(selecionados, key=lambda r: r.score, reverse=True))


ESTRATEGIAS_ORDENACAO = {
    "bordas": ordenar_bordas,
    "decrescente": ordenar_relevancia_decrescente,
    "crescente": ordenar_relevancia_crescente,
    "documento": ordenar_por_documento,
}


def reordenar(
    resultados: list[Resultado], estrategia: str = "bordas"
) -> list[Resultado]:
    """Despacha para a estratégia de ordenação escolhida."""
    if estrategia not in ESTRATEGIAS_ORDENACAO:
        raise ValueError(
            f"Estratégia '{estrategia}' desconhecida. "
            f"Disponíveis: {', '.join(ESTRATEGIAS_ORDENACAO)}"
        )
    return ESTRATEGIAS_ORDENACAO[estrategia](resultados)


def explicar_ordenacao(
    antes: list[Resultado], depois: list[Resultado]
) -> None:  # pragma: no cover - utilitário de demonstração
    """Mostra lado a lado como as passagens foram reposicionadas."""
    posicoes = {r.chunk.chunk_id: i for i, r in enumerate(antes, start=1)}
    print("\n  posição no prompt <- posição no ranking")
    for nova, resultado in enumerate(depois, start=1):
        antiga = posicoes.get(resultado.chunk.chunk_id, "?")
        print(f"    {nova:>2}  <-  {antiga:>2}   [{resultado.chunk.chunk_id}]")
