"""
Reranking: segunda etapa de ordenação sobre os candidatos recuperados.

A arquitetura de dois estágios é padrão em busca moderna: um recuperador rápido
traz os top-N de um corpus grande, e um reranker caro e preciso reordena só
esses N. O BEIR (Thakur et al., 2021) confirmou que modelos de reranking obtêm,
em média, o melhor desempenho zero-shot — ao custo computacional mais alto.

Por que funciona: o recuperador denso é um bi-encoder, que codifica consulta e
documento separadamente. Essa separação permite pré-computar os vetores do
corpus, mas impede qualquer interação entre os termos da consulta e do
documento. O reranker vê os dois juntos e pode julgar a relevância de verdade.

Sem um cross-encoder treinado, usamos um LLM de instrução como juiz. Três
formatos:

- pointwise: uma nota por passagem, chamadas independentes e paralelizáveis.
  Custo linear. Fragilidade: notas de chamadas distintas não são perfeitamente
  comparáveis entre si.
- listwise: todas as passagens numeradas em uma chamada, o modelo devolve a
  ordem. Aproveita comparação direta, mas sofre com o efeito Lost in the Middle
  dentro da própria lista.
- corte por relevância: descarta o que o modelo julgar irrelevante, em vez de
  reordenar. Inspirado no token IsREL do Self-RAG.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from config import SETTINGS, Settings, get_client
from vector_store import Resultado

PROMPT_POINTWISE = """Avalie o quanto a passagem abaixo é relevante para responder \
à pergunta.

Escala:
0-2   irrelevante: não tem relação com a pergunta
3-5   tangencial: mesmo assunto geral, mas não responde
6-8   relevante: contém parte da informação necessária
9-10  altamente relevante: responde diretamente à pergunta

Pergunta: {pergunta}

Passagem:
{passagem}

Responda APENAS com o número inteiro de 0 a 10, sem explicação."""

PROMPT_LISTWISE = """Ordene as passagens abaixo da mais relevante para a menos \
relevante em relação à pergunta.

Pergunta: {pergunta}

{passagens}

Responda APENAS com os números das passagens na nova ordem, separados por \
vírgula, da mais relevante para a menos relevante. Exemplo: 3,1,5,2,4
Inclua todos os números exatamente uma vez."""


@dataclass
class ItemRerank:
    """Um candidato com sua nota atribuída pelo reranker."""

    resultado: Resultado
    nota: float
    score_original: float


# --------------------------------------------------------------------------- #
# Pointwise
# --------------------------------------------------------------------------- #


def _extrair_nota(texto: str) -> float | None:
    """Extrai o primeiro número de 0 a 10 da resposta do modelo."""
    match = re.search(r"\b(10|[0-9])(?:[.,](\d+))?\b", texto.strip())
    if not match:
        return None
    inteiro = float(match.group(1))
    if match.group(2):
        inteiro += float(f"0.{match.group(2)}")
    return min(10.0, max(0.0, inteiro))


def rerank_pointwise(
    pergunta: str,
    candidatos: list[Resultado],
    top_k: int = 4,
    settings: Settings | None = None,
    verboso: bool = False,
) -> list[Resultado]:
    """Atribui uma nota 0-10 a cada candidato e reordena por nota."""
    settings = settings or SETTINGS
    if not candidatos:
        return []

    client = get_client(settings)
    itens: list[ItemRerank] = []

    for candidato in candidatos:
        try:
            resposta = client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT_POINTWISE.format(
                            pergunta=pergunta,
                            passagem=candidato.chunk.texto[:1500],
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=8,
            )
            nota = _extrair_nota(resposta.choices[0].message.content or "")
        except Exception as erro:  # noqa: BLE001
            print(f"  [rerank] falha em {candidato.chunk.chunk_id}: {erro}")
            nota = None

        # Sem nota utilizável, preserva a posição original de forma neutra:
        # nota 5 empata no meio da escala em vez de eliminar o candidato.
        if nota is None:
            nota = 5.0

        itens.append(ItemRerank(candidato, nota, candidato.score))
        if verboso:
            previa = candidato.chunk.texto[:60].replace("\n", " ")
            print(f"    nota={nota:>4.1f}  [{candidato.chunk.chunk_id}] {previa}...")

    # Desempate pelo score original do recuperador.
    itens.sort(key=lambda it: (it.nota, it.score_original), reverse=True)

    return [
        Resultado(chunk=it.resultado.chunk, score=it.nota / 10.0, rank=posicao)
        for posicao, it in enumerate(itens[:top_k], start=1)
    ]


# --------------------------------------------------------------------------- #
# Listwise
# --------------------------------------------------------------------------- #


def rerank_listwise(
    pergunta: str,
    candidatos: list[Resultado],
    top_k: int = 4,
    settings: Settings | None = None,
) -> list[Resultado]:
    """Envia todas as passagens numeradas e pede a nova ordem em uma chamada."""
    settings = settings or SETTINGS
    if not candidatos:
        return []
    if len(candidatos) == 1:
        return candidatos[:top_k]

    blocos = "\n\n".join(
        f"[{i}] {c.chunk.texto[:700]}" for i, c in enumerate(candidatos, start=1)
    )

    try:
        client = get_client(settings)
        resposta = client.chat.completions.create(
            model=settings.chat_model,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT_LISTWISE.format(
                        pergunta=pergunta, passagens=blocos
                    ),
                }
            ],
            temperature=0.0,
            max_tokens=100,
        )
        texto = resposta.choices[0].message.content or ""
    except Exception as erro:  # noqa: BLE001
        print(f"  [rerank listwise] falha ({erro}); mantendo a ordem original")
        return candidatos[:top_k]

    # Interpreta a ordem devolvida, ignorando números fora do intervalo.
    numeros = [int(n) for n in re.findall(r"\d+", texto)]
    ordem: list[int] = []
    for n in numeros:
        if 1 <= n <= len(candidatos) and n not in ordem:
            ordem.append(n)

    # Completa com quem o modelo esqueceu, preservando a ordem original.
    for i in range(1, len(candidatos) + 1):
        if i not in ordem:
            ordem.append(i)

    return [
        Resultado(
            chunk=candidatos[n - 1].chunk,
            score=candidatos[n - 1].score,
            rank=posicao,
        )
        for posicao, n in enumerate(ordem[:top_k], start=1)
    ]


# --------------------------------------------------------------------------- #
# Filtro de relevância (IsREL do Self-RAG)
# --------------------------------------------------------------------------- #

PROMPT_ISREL = """A passagem abaixo fornece informação útil para responder à pergunta?

Pergunta: {pergunta}

Passagem:
{passagem}

Responda APENAS com um JSON: {{"relevante": true}} ou {{"relevante": false}}"""


def filtrar_relevantes(
    pergunta: str,
    candidatos: list[Resultado],
    settings: Settings | None = None,
) -> tuple[list[Resultado], list[Resultado]]:
    """
    Separa candidatos relevantes dos irrelevantes (token IsREL do Self-RAG).

    Devolve (relevantes, descartados). Se todos forem descartados, devolve os
    originais — um filtro que zera o contexto é pior do que um contexto ruidoso.
    """
    settings = settings or SETTINGS
    if not candidatos:
        return [], []

    client = get_client(settings)
    relevantes: list[Resultado] = []
    descartados: list[Resultado] = []

    for candidato in candidatos:
        veredito = True
        try:
            resposta = client.chat.completions.create(
                model=settings.chat_model,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT_ISREL.format(
                            pergunta=pergunta,
                            passagem=candidato.chunk.texto[:1500],
                        ),
                    }
                ],
                temperature=0.0,
                max_tokens=20,
            )
            texto = (resposta.choices[0].message.content or "").strip()
            match = re.search(r"\{.*\}", texto, re.DOTALL)
            if match:
                veredito = bool(json.loads(match.group(0)).get("relevante", True))
            else:
                veredito = "true" in texto.lower()
        except Exception as erro:  # noqa: BLE001
            print(f"  [IsREL] falha em {candidato.chunk.chunk_id}: {erro}")

        (relevantes if veredito else descartados).append(candidato)

    if not relevantes:
        print("  [IsREL] nenhum candidato passou no filtro; mantendo todos")
        return candidatos, []

    relevantes = [
        Resultado(chunk=r.chunk, score=r.score, rank=i)
        for i, r in enumerate(relevantes, start=1)
    ]
    return relevantes, descartados


ESTRATEGIAS_RERANK = {
    "pointwise": rerank_pointwise,
    "listwise": rerank_listwise,
}


def rerank(
    pergunta: str,
    candidatos: list[Resultado],
    top_k: int = 4,
    estrategia: str = "pointwise",
    settings: Settings | None = None,
) -> list[Resultado]:
    """Despacha para a estratégia de reranking escolhida."""
    if estrategia not in ESTRATEGIAS_RERANK:
        raise ValueError(
            f"Estratégia '{estrategia}' desconhecida. "
            f"Disponíveis: {', '.join(ESTRATEGIAS_RERANK)}"
        )
    return ESTRATEGIAS_RERANK[estrategia](pergunta, candidatos, top_k, settings)
