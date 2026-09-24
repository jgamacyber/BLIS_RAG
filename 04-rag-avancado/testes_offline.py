"""
Testes offline — validam o pipeline sem chamar a API nem gastar créditos.

Cobrem tudo que não depende do LLM: chunking, BM25, fusão RRF, ordenação de
contexto, k-means do RAPTOR, vector store e métricas de avaliação.

Rode antes de gastar créditos, para garantir que o encanamento está correto:

    python testes_offline.py
"""

from __future__ import annotations

import sys

from bm25 import BM25, tokenizar
from chunking import Chunk, dividir_documento, estatisticas
from context_ordering import ordenar_bordas, reordenar, truncar_contexto
from evaluate import hit_e_mrr, ndcg, recall_precision
from embeddings import LocalHashingEmbedder
from fusao import reciprocal_rank_fusion, soma_ponderada
from raptor import kmeans
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


def testar_bm25() -> None:
    secao("2. BM25")
    chunks = [
        _chunk("O BM25 usa IDF e saturação de frequência de termo.", "c1", "bm25"),
        _chunk("Embeddings densos capturam sinônimos e paráfrases.", "c2", "denso"),
        _chunk("O DPR treina um bi-encoder com negativos no lote.", "c3", "dpr"),
        _chunk("A recuperação densa usa produto interno entre vetores.", "c4", "denso"),
    ]
    indice = BM25(chunks)

    checar(len(indice) == 4, "índice contém todos os documentos")
    checar(indice.avgdl > 0, "comprimento médio calculado")

    resultados = indice.buscar("Como funciona o BM25 e o IDF?", top_k=2)
    checar(bool(resultados), "busca devolve resultados")
    checar(resultados[0].chunk.chunk_id == "c1", "o chunk sobre BM25 vem em primeiro")
    checar(
        all(resultados[i].score >= resultados[i + 1].score for i in range(len(resultados) - 1)),
        "scores em ordem decrescente",
    )
    checar(
        all(r.rank == i + 1 for i, r in enumerate(resultados)),
        "ranks numerados a partir de 1",
    )

    checar(tokenizar("A função de ranqueamento") == ["funcao", "ranqueamento"],
           "tokenizador remove acentos e stopwords")
    checar(indice.buscar("xyzabc inexistente", top_k=3) == [],
           "consulta sem termos no corpus devolve lista vazia")
    checar(indice.buscar("de a o que", top_k=3) == [],
           "consulta só com stopwords devolve lista vazia")


def testar_fusao() -> None:
    secao("3. Fusão de rankings (RRF e soma ponderada)")
    denso = [_resultado("a", 0.9, 1), _resultado("b", 0.8, 2), _resultado("c", 0.7, 3)]
    lexical = [_resultado("c", 12.0, 1), _resultado("d", 8.0, 2), _resultado("a", 5.0, 3)]

    fundido = reciprocal_rank_fusion([denso, lexical], k=60, top_k=4)
    ids = [r.chunk.chunk_id for r in fundido]

    checar(len(fundido) == 4, "RRF devolve a união dos rankings")
    checar(ids[0] == "a", "documento presente em ambos (1º e 3º) lidera o RRF")
    checar("d" in ids, "documento exclusivo do lexical entra no resultado")
    checar(
        all(fundido[i].score >= fundido[i + 1].score for i in range(len(fundido) - 1)),
        "RRF devolve em ordem decrescente",
    )
    checar(
        all(r.rank == i + 1 for i, r in enumerate(fundido)),
        "RRF renumera os ranks",
    )

    # Verificação numérica: 'a' aparece em rank 1 e rank 3 -> 1/61 + 1/63
    esperado = 1 / 61 + 1 / 63
    checar(abs(fundido[0].score - esperado) < 1e-9, "score do RRF confere com a fórmula")

    ponderada = soma_ponderada(denso, lexical, peso_denso=0.5, top_k=4)
    checar(len(ponderada) == 4, "soma ponderada devolve a união")
    checar(
        all(0.0 <= r.score <= 1.0 for r in ponderada),
        "scores da soma ponderada ficam normalizados em [0,1]",
    )

    checar(reciprocal_rank_fusion([], top_k=3) == [], "RRF com lista vazia não quebra")


def testar_ordenacao() -> None:
    secao("4. Ordenação de contexto (Lost in the Middle)")
    ranking = [_resultado(str(i), 1.0 - i / 10, i) for i in range(1, 8)]

    bordas = ordenar_bordas(ranking)
    ids = [r.chunk.chunk_id for r in bordas]

    checar(len(bordas) == 7, "ordenação preserva todas as passagens")
    checar(ids[0] == "1", "a mais relevante fica na primeira posição")
    checar(ids[-1] == "2", "a segunda mais relevante fica na última posição")
    checar(ids[len(ids) // 2] == "7", "a menos relevante fica no meio (zona morta)")
    checar(
        all(r.rank == i + 1 for i, r in enumerate(bordas)),
        "ranks renumerados conforme a posição no prompt",
    )
    checar(set(ids) == {str(i) for i in range(1, 8)}, "nenhuma passagem some ou duplica")

    crescente = reordenar(ranking, "crescente")
    checar(
        crescente[-1].chunk.chunk_id == "1",
        "ordem crescente deixa a mais relevante por último (recência)",
    )

    decrescente = reordenar(ranking, "decrescente")
    checar(decrescente[0].chunk.chunk_id == "1", "ordem decrescente preserva o ranking")

    checar(len(ordenar_bordas(ranking[:2])) == 2, "listas curtas passam intactas")

    longos = [
        Resultado(chunk=_chunk("x" * 500, str(i)), score=1.0 - i / 10, rank=i)
        for i in range(1, 6)
    ]
    truncados = truncar_contexto(longos, max_chars=1200)
    checar(len(truncados) == 2, "truncamento respeita o orçamento de caracteres")
    checar(truncados[0].chunk.chunk_id == "1", "truncamento preserva os mais relevantes")


def testar_kmeans() -> None:
    secao("5. K-means do RAPTOR")
    # Dois grupos bem separados no espaço.
    grupo_a = [[1.0, 0.0, 0.0], [0.98, 0.05, 0.0], [0.95, 0.1, 0.02]]
    grupo_b = [[0.0, 0.0, 1.0], [0.02, 0.05, 0.98], [0.0, 0.1, 0.95]]
    rotulos = kmeans(grupo_a + grupo_b, k=2)

    checar(len(rotulos) == 6, "um rótulo por vetor")
    checar(len(set(rotulos)) == 2, "encontra exatamente 2 clusters")
    checar(
        len(set(rotulos[:3])) == 1 and len(set(rotulos[3:])) == 1,
        "separa corretamente os dois grupos",
    )
    checar(rotulos[0] != rotulos[3], "os grupos recebem rótulos diferentes")
    checar(kmeans([], k=3) == [], "k-means com entrada vazia não quebra")
    checar(kmeans(grupo_a, k=10) is not None, "k maior que n não quebra")
    checar(
        kmeans(grupo_a + grupo_b, k=2) == rotulos,
        "resultado é determinístico (mesma semente)",
    )


def testar_vector_store() -> None:
    secao("6. Vector store e embeddings locais")
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
    secao("7. Métricas de avaliação")
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
    testar_bm25()
    testar_fusao()
    testar_ordenacao()
    testar_kmeans()
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
