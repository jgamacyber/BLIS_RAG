"""
RAPTOR — árvore de resumos recursivos (versão simplificada).

Implementação inspirada em Sarthi et al. (ICLR 2024), "RAPTOR: Recursive
Abstractive Processing for Tree-Organized Retrieval".

A ideia: recuperar apenas trechos curtos e contíguos impede responder perguntas
temáticas, cuja resposta está distribuída ao longo do texto. O RAPTOR constrói
uma árvore de baixo para cima: agrupa chunks semanticamente próximos, resume
cada grupo com um LLM, re-embedda os resumos e repete. Os nós internos passam a
conter informação em níveis crescentes de abstração.

Na consulta, a estratégia de "árvore colapsada" busca em TODOS os nós de TODAS
as camadas simultaneamente. Perguntas específicas casam com folhas; perguntas
temáticas casam com resumos de alto nível.

SIMPLIFICAÇÕES em relação ao paper:
- O paper usa UMAP para redução de dimensionalidade + Gaussian Mixture Models
  com clusterização suave (um nó pode pertencer a vários clusters), escolhendo
  o número de clusters por BIC. Aqui usamos k-means simples (implementado neste
  arquivo, sem scikit-learn) com clusterização rígida.
- O paper embedda com SBERT; aqui reusamos o embedder configurado.
Essas escolhas mantêm o arquivo sem dependências pesadas e o algoritmo legível;
o comportamento qualitativo (abstração crescente) é preservado.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from chunking import Chunk
from config import SETTINGS, Settings, get_client
from embeddings import Embedder

PROMPT_RESUMO = """Escreva um resumo dos trechos abaixo.

O resumo deve:
- preservar os fatos, números, nomes e termos técnicos importantes;
- integrar os trechos em um texto coerente, não listá-los separadamente;
- ter no máximo {max_palavras} palavras.

Responda apenas com o resumo.

TRECHOS:
{trechos}"""


@dataclass
class NoRaptor:
    """Um nó da árvore: folha (chunk original) ou resumo de um cluster."""

    chunk: Chunk
    nivel: int = 0
    filhos: list[str] = field(default_factory=list)

    @property
    def eh_folha(self) -> bool:
        return self.nivel == 0


# --------------------------------------------------------------------------- #
# K-means (sem scikit-learn)
# --------------------------------------------------------------------------- #


def _distancia_cosseno(a: list[float], b: list[float]) -> float:
    produto = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 1.0
    return 1.0 - produto / (na * nb)


def _media(vetores: list[list[float]]) -> list[float]:
    n = len(vetores)
    dim = len(vetores[0])
    return [sum(v[i] for v in vetores) / n for i in range(dim)]


def kmeans(
    vetores: list[list[float]],
    k: int,
    max_iter: int = 30,
    semente: int = 42,
) -> list[int]:
    """
    K-means com distância de cosseno e inicialização k-means++ simplificada.

    Devolve a lista de rótulos de cluster, um por vetor.
    """
    n = len(vetores)
    if n == 0:
        return []
    k = max(1, min(k, n))
    if k == 1:
        return [0] * n

    rng = random.Random(semente)

    # k-means++: o primeiro centro é aleatório; os demais são escolhidos com
    # probabilidade proporcional à distância ao centro mais próximo.
    centros = [list(vetores[rng.randrange(n)])]
    while len(centros) < k:
        distancias = [
            min(_distancia_cosseno(v, c) for c in centros) ** 2 for v in vetores
        ]
        total = sum(distancias)
        if total == 0:
            centros.append(list(vetores[rng.randrange(n)]))
            continue
        alvo = rng.random() * total
        acumulado = 0.0
        for i, d in enumerate(distancias):
            acumulado += d
            if acumulado >= alvo:
                centros.append(list(vetores[i]))
                break

    rotulos = [0] * n
    for _ in range(max_iter):
        mudou = False
        for i, vetor in enumerate(vetores):
            novo = min(
                range(len(centros)), key=lambda c: _distancia_cosseno(vetor, centros[c])
            )
            if novo != rotulos[i]:
                rotulos[i] = novo
                mudou = True

        for c in range(len(centros)):
            membros = [vetores[i] for i in range(n) if rotulos[i] == c]
            if membros:
                centros[c] = _media(membros)

        if not mudou:
            break

    return rotulos


# --------------------------------------------------------------------------- #
# Construção da árvore
# --------------------------------------------------------------------------- #


def resumir_cluster(
    textos: list[str], max_palavras: int = 200, settings: Settings | None = None
) -> str:
    """Chama o LLM para resumir os textos de um cluster."""
    settings = settings or SETTINGS
    trechos = "\n\n---\n\n".join(t[:1500] for t in textos)

    client = get_client(settings)
    resposta = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {
                "role": "user",
                "content": PROMPT_RESUMO.format(
                    max_palavras=max_palavras, trechos=trechos[:12000]
                ),
            }
        ],
        temperature=0.0,
        max_tokens=int(max_palavras * 2.5),
    )
    return (resposta.choices[0].message.content or "").strip()


def construir_arvore(
    chunks: list[Chunk],
    embedder: Embedder,
    niveis: int = 2,
    tamanho_cluster: int = 5,
    settings: Settings | None = None,
    verboso: bool = True,
) -> list[NoRaptor]:
    """
    Constrói a árvore RAPTOR de baixo para cima.

    Devolve TODOS os nós — folhas e resumos —, prontos para a estratégia de
    árvore colapsada, em que a busca percorre todas as camadas de uma vez.
    """
    settings = settings or SETTINGS

    nos = [NoRaptor(chunk=c, nivel=0) for c in chunks]
    camada_atual = list(nos)

    if verboso:
        print(f"  nível 0 (folhas): {len(camada_atual)} nós")

    for nivel in range(1, niveis + 1):
        if len(camada_atual) <= tamanho_cluster:
            if verboso:
                print(
                    f"  nível {nivel}: apenas {len(camada_atual)} nós restantes; "
                    "parando a recursão"
                )
            break

        vetores = embedder.embed([no.chunk.texto for no in camada_atual])
        k = max(2, len(camada_atual) // tamanho_cluster)
        rotulos = kmeans(vetores, k)

        grupos: dict[int, list[NoRaptor]] = {}
        for rotulo, no in zip(rotulos, camada_atual):
            grupos.setdefault(rotulo, []).append(no)

        if verboso:
            print(f"  nível {nivel}: {len(camada_atual)} nós -> {len(grupos)} clusters")

        nova_camada: list[NoRaptor] = []
        for indice, (rotulo, membros) in enumerate(sorted(grupos.items())):
            if len(membros) < 2:
                # Cluster unitário não gera abstração nova; sobe como está.
                nova_camada.append(membros[0])
                continue

            try:
                resumo = resumir_cluster(
                    [m.chunk.texto for m in membros], settings=settings
                )
            except Exception as erro:  # noqa: BLE001
                print(f"    [raptor] falha ao resumir cluster {rotulo}: {erro}")
                continue

            if not resumo:
                continue

            docs = sorted({m.chunk.doc_id for m in membros})
            no_resumo = NoRaptor(
                chunk=Chunk(
                    texto=resumo,
                    doc_id=docs[0] if len(docs) == 1 else "+".join(docs[:3]),
                    chunk_id=f"resumo_n{nivel}_{indice:02d}",
                    titulo=f"Resumo nível {nivel} ({len(membros)} trechos)",
                    posicao=indice,
                    metadados={
                        "tipo": "resumo_raptor",
                        "nivel": nivel,
                        "documentos": docs,
                    },
                ),
                nivel=nivel,
                filhos=[m.chunk.chunk_id for m in membros],
            )
            nova_camada.append(no_resumo)
            nos.append(no_resumo)

            if verboso:
                previa = resumo[:80].replace("\n", " ")
                print(f"    cluster {indice:>2} ({len(membros)} nós): {previa}...")

        if not nova_camada:
            break
        camada_atual = nova_camada

    if verboso:
        por_nivel: dict[int, int] = {}
        for no in nos:
            por_nivel[no.nivel] = por_nivel.get(no.nivel, 0) + 1
        distribuicao = " | ".join(f"n{n}={q}" for n, q in sorted(por_nivel.items()))
        print(f"  árvore pronta: {len(nos)} nós no total ({distribuicao})")

    return nos


def chunks_da_arvore(nos: list[NoRaptor]) -> list[Chunk]:
    """Extrai os chunks de todos os nós, para indexação em árvore colapsada."""
    return [no.chunk for no in nos]
