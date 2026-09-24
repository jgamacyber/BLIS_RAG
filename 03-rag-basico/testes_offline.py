"""
Testes offline — validam o pipeline sem chamar a API nem gastar créditos.

Cobrem tudo que não depende do LLM: chunking, vector store, embeddings locais
e métricas de avaliação.

Rode antes de gastar créditos, para garantir que o encanamento está correto:

    python testes_offline.py
"""

from __future__ import annotations

import sys

from chunking import Chunk, dividir_documento, estatisticas
from evaluate import hit_e_mrr, ndcg, recall_precision
from embeddings import LocalHashingEmbedder
from vector_store import Resultado, VectorStore, similaridade_cosseno

FALHAS: list[str] = []


def checar(condicao: bool, descricao: str) -> None:
    if condicao:
        print(f"  [ok]   {descricao}")
    else:
        print(f"  [FALHA] {descricao}")
        FALHAS.append(descricao)


def secao(titulo: str) -> None:
    print(f"\n{titulo}")
    print("-" * 62)


def _chunk(texto: str, chunk_id: str, doc_id: str = "doc") -> Chunk:
    return Chunk(texto=texto, doc_id=doc_id, chunk_id=chunk_id)


def _resultado(chunk_id: str, score: float, rank: int, doc="doc") -> Resultado:
    return Resultado(chunk=_chunk(f"texto {chunk_id}", chunk_id, doc), score=score, rank=rank)


# --------------------------------------------------------------------------- #

def testar_chunking() -> None:
    secao("1. Chunking")
    texto = (
        "Primeiro parágrafo com uma frase. E outra frase aqui.\n\n"
        "Segundo parágrafo, bem mais longo, escrito para forçar a divisão do "
        "texto em mais de um pedaço quando o orçamento de caracteres for "
        "pequeno o suficiente para provocar isso. " * 4
    )

    for estrategia in ("fixo", "frases", "recursivo"):
        chunks = dividir_documento(
            texto, "doc_teste", estrategia=estrategia, chunk_size=200, overlap=40
        )
        checar(len(chunks) > 1, f"'{estrategia}' produz múltiplos chunks ({len(chunks)})")
        checar(
            all(c.texto.strip() for c in chunks),
            f"'{estrategia}' não produz chunks vazios",
        )
        checar(
            len({c.chunk_id for c in chunks}) == len(chunks),
            f"'{estrategia}' gera chunk_ids únicos",
        )

    frases = dividir_documento(texto, "d", estrategia="frases", chunk_size=200, overlap=0)
    checar(
        all(not c.texto.endswith(" ") for c in frases),
        "'frases' não deixa espaço solto no fim",
    )

    st = estatisticas(frases)
    checar(st["n_chunks"] == len(frases), "estatísticas contam os chunks corretamente")
    checar(dividir_documento("", "vazio") == [], "documento vazio gera zero chunks")






def testar_vector_store() -> None:
    secao("2. Vector store e embeddings locais")
    embedder = LocalHashingEmbedder(dimensao=128)
    chunks = [
        _chunk("recuperação densa com embeddings e produto interno", "c1"),
        _chunk("bolo de cenoura com cobertura de chocolate", "c2"),
        _chunk("busca vetorial usando similaridade de cosseno", "c3"),
    ]
    vetores = embedder.embed([c.texto for c in chunks])

    checar(len(vetores) == 3, "gera um vetor por texto")
    checar(all(len(v) == 128 for v in vetores), "dimensão correta")
    checar(
        abs(sum(x * x for x in vetores[0]) ** 0.5 - 1.0) < 1e-6,
        "vetores normalizados (norma L2 = 1)",
    )
    checar(
        embedder.embed(["teste"]) == embedder.embed(["teste"]),
        "embedder local é determinístico",
    )

    store = VectorStore(nome_modelo=embedder.nome)
    store.adicionar(chunks, vetores)
    checar(len(store) == 3, "store contém todos os chunks")

    consulta = embedder.embed_um("busca vetorial por similaridade")
    resultados = store.buscar(consulta, top_k=2)
    checar(len(resultados) == 2, "busca respeita o top_k")
    checar(
        resultados[0].chunk.chunk_id in {"c1", "c3"},
        "recupera um chunk do assunto certo, não a receita de bolo",
    )
    checar(store.buscar(consulta, top_k=99)[0].rank == 1, "top_k maior que n não quebra")

    checar(abs(similaridade_cosseno([1, 0], [1, 0]) - 1.0) < 1e-9, "cosseno de iguais = 1")
    checar(abs(similaridade_cosseno([1, 0], [0, 1])) < 1e-9, "cosseno de ortogonais = 0")
    checar(similaridade_cosseno([0, 0], [1, 1]) == 0.0, "vetor nulo não divide por zero")


def testar_metricas() -> None:
    secao("3. Métricas de avaliação")
    relevantes = {"x", "y"}

    hit, mrr = hit_e_mrr(["a", "x", "b"], relevantes)
    checar(hit == 1.0 and abs(mrr - 0.5) < 1e-9, "hit e MRR com acerto na 2ª posição")

    hit, mrr = hit_e_mrr(["a", "b"], relevantes)
    checar(hit == 0.0 and mrr == 0.0, "sem acerto, hit e MRR zeram")

    recall, precision = recall_precision(["x", "a", "y", "b"], relevantes, k=4)
    checar(abs(recall - 1.0) < 1e-9, "recall de 100% com os dois relevantes")
    checar(abs(precision - 0.5) < 1e-9, "precision de 50% com 2 de 4")

    perfeito = ndcg(["x", "y", "a"], relevantes, k=3)
    pior = ndcg(["a", "x", "y"], relevantes, k=3)
    checar(abs(perfeito - 1.0) < 1e-9, "nDCG do ranking perfeito é 1.0")
    checar(pior < perfeito, "nDCG penaliza relevantes em posições baixas")
    checar(ndcg(["a", "b"], relevantes, k=2) == 0.0, "nDCG sem acertos é 0")


def main() -> int:
    print("=" * 62)
    print("  TESTES OFFLINE — sem API, sem custo")
    print("=" * 62)

    testar_chunking()
    testar_vector_store()
    testar_metricas()

    print("\n" + "=" * 62)
    if FALHAS:
        print(f"  {len(FALHAS)} FALHA(S):")
        for f in FALHAS:
            print(f"    - {f}")
        print("=" * 62)
        return 1

    print("  Todos os testes passaram.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
