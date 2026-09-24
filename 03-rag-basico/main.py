"""
BLIS RAG — Módulo 03: RAG Básico
CLI para indexar o corpus, consultar e avaliar o retriever.

Uso:
    python main.py indexar                      # indexa data/corpus
    python main.py perguntar "sua pergunta"     # consulta o índice
    python main.py chat                         # modo interativo
    python main.py avaliar                      # métricas do retriever
    python main.py comparar-chunking            # compara estratégias
    python main.py demo                         # roteiro completo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import SETTINGS, resumo_config
from evaluate import avaliar_retriever, carregar_conjunto, comparar
from ingest import ingerir
from rag_pipeline import CAMINHO_INDICE, RAGPipeline

DIR_CORPUS = Path("data/corpus")
LARGURA = 70


def banner(titulo: str) -> None:
    print("\n" + "=" * LARGURA)
    print(f"  {titulo}")
    print("=" * LARGURA)
    print(f"  {resumo_config()}")


def _exigir_indice() -> RAGPipeline:
    if not CAMINHO_INDICE.exists():
        print(f"Índice não encontrado em {CAMINHO_INDICE}.")
        print("Rode primeiro:  python main.py indexar")
        sys.exit(1)
    return RAGPipeline.carregar(CAMINHO_INDICE)


# --------------------------------------------------------------------------- #
# Comandos
# --------------------------------------------------------------------------- #


def cmd_indexar(args: argparse.Namespace) -> None:
    banner("INDEXAÇÃO DO CORPUS")
    pipeline = RAGPipeline()
    pipeline.indexar_diretorio(args.corpus, estrategia=args.estrategia)
    pipeline.salvar(CAMINHO_INDICE)


def cmd_perguntar(args: argparse.Namespace) -> None:
    banner("CONSULTA")
    pipeline = _exigir_indice()
    resposta = pipeline.perguntar(args.pergunta, top_k=args.top_k)
    resposta.imprimir(mostrar_contexto=not args.sem_contexto)


def cmd_chat(args: argparse.Namespace) -> None:
    banner("MODO INTERATIVO  (ENTER vazio ou 'sair' para encerrar)")
    pipeline = _exigir_indice()
    print(f"  {pipeline.store.resumo()}")

    while True:
        try:
            pergunta = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not pergunta or pergunta.lower() in {"sair", "exit", "quit"}:
            break
        try:
            pipeline.perguntar(pergunta, top_k=args.top_k).imprimir()
        except Exception as erro:  # noqa: BLE001
            print(f"  [erro] {erro}")
    print("Encerrado.")


def cmd_avaliar(args: argparse.Namespace) -> None:
    banner("AVALIAÇÃO DO RETRIEVER")
    pipeline = _exigir_indice()
    conjunto = carregar_conjunto()
    print(f"  {len(conjunto)} consultas de avaliação | top_k={args.top_k}\n")

    metricas = avaliar_retriever(
        lambda p, k: pipeline.recuperar(p, top_k=k),
        conjunto,
        k=args.top_k,
    )
    metricas.imprimir("Retriever denso (embeddings)")


def cmd_comparar_chunking(args: argparse.Namespace) -> None:
    banner("COMPARAÇÃO DE ESTRATÉGIAS DE CHUNKING")
    conjunto = carregar_conjunto()
    resultados = {}

    for estrategia in ("fixo", "frases", "recursivo"):
        print(f"\n--- estratégia: {estrategia} ---")
        pipeline = RAGPipeline()
        chunks = ingerir(
            args.corpus,
            estrategia=estrategia,
            chunk_size=SETTINGS.chunk_size,
            overlap=SETTINGS.chunk_overlap,
            verboso=False,
        )
        pipeline.indexar_chunks(chunks, verboso=False)
        print(f"  {len(chunks)} chunks indexados")

        resultados[estrategia] = avaliar_retriever(
            lambda p, k: pipeline.recuperar(p, top_k=k),
            conjunto,
            k=args.top_k,
            verboso=False,
        )

    comparar(resultados)
    print(
        "\nObservação: com um corpus pequeno as diferenças são modestas. "
        "O efeito do chunking cresce com o tamanho e a heterogeneidade do corpus."
    )


def cmd_demo(args: argparse.Namespace) -> None:
    banner("DEMONSTRAÇÃO COMPLETA — RAG BÁSICO")
    pipeline = RAGPipeline()
    pipeline.indexar_diretorio(args.corpus)
    pipeline.salvar(CAMINHO_INDICE)

    perguntas = [
        "O que é RAG e quem propôs a arquitetura?",
        "Qual a diferença entre RAG-Sequence e RAG-Token?",
        "Por que embeddings densos superam o BM25 em alguns casos?",
        "Qual é a capital da Mongólia?",  # fora do corpus: deve haver abstenção
    ]

    for pergunta in perguntas:
        pipeline.perguntar(pergunta).imprimir()
        print("\n" + "-" * LARGURA)

    print(
        "\nNote a última pergunta: como o corpus não cobre o assunto, o sistema "
        "deve se abster em vez de inventar. Essa é a diferença entre um RAG "
        "honesto e um gerador de alucinações com citações."
    )


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BLIS RAG — Módulo 03: RAG Básico",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    def add_corpus(p):
        p.add_argument("--corpus", default=str(DIR_CORPUS), help="diretório do corpus")

    def add_topk(p):
        p.add_argument(
            "--top-k", type=int, default=SETTINGS.top_k, help="passagens recuperadas"
        )

    p = sub.add_parser("indexar", help="indexa o corpus")
    add_corpus(p)
    p.add_argument(
        "--estrategia",
        default="recursivo",
        choices=["fixo", "frases", "recursivo"],
        help="estratégia de chunking",
    )
    p.set_defaults(func=cmd_indexar)

    p = sub.add_parser("perguntar", help="faz uma pergunta ao índice")
    p.add_argument("pergunta")
    add_topk(p)
    p.add_argument("--sem-contexto", action="store_true", help="oculta as passagens")
    p.set_defaults(func=cmd_perguntar)

    p = sub.add_parser("chat", help="modo interativo")
    add_topk(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("avaliar", help="métricas do retriever")
    add_topk(p)
    p.set_defaults(func=cmd_avaliar)

    p = sub.add_parser("comparar-chunking", help="compara estratégias de chunking")
    add_corpus(p)
    add_topk(p)
    p.set_defaults(func=cmd_comparar_chunking)

    p = sub.add_parser("demo", help="roteiro completo de demonstração")
    add_corpus(p)
    p.set_defaults(func=cmd_demo)

    return parser


def main() -> None:
    args = construir_parser().parse_args()
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\nInterrompido.")
        sys.exit(130)
    except Exception as erro:  # noqa: BLE001
        print(f"\n[erro] {erro}")
        sys.exit(1)


if __name__ == "__main__":
    main()
