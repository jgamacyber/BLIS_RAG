"""
BLIS RAG — Módulo 04: RAG Avançado
CLI para indexar, consultar e comparar as técnicas avançadas.

Uso:
    python main.py indexar                       # indexa data/corpus
    python main.py indexar --raptor              # indexa com árvore RAPTOR
    python main.py perguntar "pergunta"          # com todas as técnicas
    python main.py perguntar "p" --preset basico # sem melhoria alguma
    python main.py chat                          # modo interativo
    python main.py comparar                      # ablação: mede cada técnica
    python main.py demo                          # roteiro guiado
    python main.py bm25 "pergunta"               # inspeciona a busca lexical
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from advanced_pipeline import CAMINHO_INDICE, AdvancedRAGPipeline, Config
from config import SETTINGS, resumo_config
from evaluate import avaliar_retriever, carregar_conjunto, comparar

DIR_CORPUS = Path("data/corpus")
LARGURA = 70

PRESETS = {
    "basico": Config.basico,
    "hibrido": lambda: Config(hibrido=True, ordenacao="bordas"),
    "hyde": lambda: Config(hibrido=True, hyde=True, ordenacao="bordas"),
    "rerank": lambda: Config(hibrido=True, reranking="pointwise", ordenacao="bordas"),
    "completo": Config.completo,
}


def banner(titulo: str) -> None:
    print("\n" + "=" * LARGURA)
    print(f"  {titulo}")
    print("=" * LARGURA)
    print(f"  {resumo_config()}")


def _exigir_indice(config: Config | None = None) -> AdvancedRAGPipeline:
    if not CAMINHO_INDICE.exists():
        print(f"Índice não encontrado em {CAMINHO_INDICE}.")
        print("Rode primeiro:  python main.py indexar")
        sys.exit(1)
    return AdvancedRAGPipeline.carregar(CAMINHO_INDICE, config=config)


# --------------------------------------------------------------------------- #
# Comandos
# --------------------------------------------------------------------------- #


def cmd_indexar(args: argparse.Namespace) -> None:
    banner("INDEXAÇÃO" + (" COM RAPTOR" if args.raptor else ""))
    if args.raptor:
        print(
            "  Atenção: o RAPTOR gera resumos com o LLM, o que consome créditos "
            "e leva alguns minutos."
        )
    pipeline = AdvancedRAGPipeline()
    pipeline.indexar_diretorio(args.corpus, usar_raptor=args.raptor)
    pipeline.salvar(CAMINHO_INDICE)


def cmd_perguntar(args: argparse.Namespace) -> None:
    config = PRESETS[args.preset]()
    banner(f"CONSULTA — preset '{args.preset}'")
    print(f"  técnicas: {', '.join(config.ativas())}")

    pipeline = _exigir_indice(config)
    resposta = pipeline.perguntar(args.pergunta, top_k=args.top_k, verboso=args.verboso)
    resposta.imprimir(mostrar_contexto=not args.sem_contexto)


def cmd_chat(args: argparse.Namespace) -> None:
    config = PRESETS[args.preset]()
    banner(f"MODO INTERATIVO — preset '{args.preset}'  (ENTER vazio para sair)")
    print(f"  técnicas: {', '.join(config.ativas())}")

    pipeline = _exigir_indice(config)
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


def cmd_comparar(args: argparse.Namespace) -> None:
    """
    Ablação: mede o retriever sob cada configuração, acumulando as técnicas.

    Compara apenas a etapa de RECUPERAÇÃO, que é o que essas métricas medem.
    Técnicas de geração (IsSUP, IsUSE, ordenação de contexto) não aparecem aqui
    porque não mudam quais documentos são recuperados — só como são usados.
    """
    banner("ABLAÇÃO — CONTRIBUIÇÃO DE CADA TÉCNICA")
    conjunto = carregar_conjunto()
    print(f"  {len(conjunto)} consultas | top_k={args.top_k}")

    configuracoes: dict[str, Config] = {
        "1. denso puro (base)": Config(hibrido=False, reranking=""),
        "2. + BM25 híbrido (RRF)": Config(hibrido=True, fusao="rrf"),
    }
    if not args.rapido:
        configuracoes["3. + reescrita multi-query"] = Config(
            hibrido=True, reescrita="multi_query"
        )
        configuracoes["4. + HyDE"] = Config(
            hibrido=True, reescrita="multi_query", hyde=True
        )
        configuracoes["5. + reranking"] = Config(
            hibrido=True, reescrita="multi_query", hyde=True, reranking="pointwise"
        )

    if args.rapido:
        print("  modo rápido: só as técnicas que não chamam o LLM\n")
    else:
        print("  ATENÇÃO: as configurações 3-5 chamam o LLM e consomem créditos.\n")

    pipeline = _exigir_indice()
    resultados = {}

    for nome, config in configuracoes.items():
        print(f"--- {nome} ---")
        pipeline.config = config
        resultados[nome] = avaliar_retriever(
            lambda p, k: pipeline.recuperar(p, top_k=k),
            conjunto,
            k=args.top_k,
            verboso=False,
        )
        m = resultados[nome]
        print(f"    Hit@{args.top_k}={m.hit_rate:.1%}  nDCG={m.ndcg:.3f}\n")

    comparar(resultados)
    print(
        "\nLeitura: cada linha acumula a técnica da linha anterior. Um ganho "
        "pequeno ou negativo indica que a técnica não ajuda NESTE corpus — o que "
        "é um resultado legítimo e vale registrar."
    )


def cmd_bm25(args: argparse.Namespace) -> None:
    banner("INSPEÇÃO DA BUSCA LEXICAL (BM25)")
    pipeline = _exigir_indice()
    bm25 = pipeline._garantir_bm25()
    print(f"  {bm25.resumo()}\n")

    print("Termos da consulta, do mais raro ao mais comum (IDF):")
    for termo, idf in bm25.termos_da_consulta(args.pergunta):
        marca = "  (ausente do corpus)" if idf == 0 else ""
        print(f"  {idf:>6.3f}  {termo}{marca}")

    print(f"\nTop-{args.top_k} por BM25:")
    for r in bm25.buscar(args.pergunta, top_k=args.top_k):
        previa = r.chunk.texto[:100].replace("\n", " ")
        print(f"  {r.rank}. [{r.chunk.chunk_id}] score={r.score:.3f}")
        print(f"     {previa}...")


def cmd_demo(args: argparse.Namespace) -> None:
    banner("DEMONSTRAÇÃO — RAG AVANÇADO")
    pipeline = AdvancedRAGPipeline()
    pipeline.indexar_diretorio(args.corpus)
    pipeline.salvar(CAMINHO_INDICE)

    pergunta = "Como o HyDE constrói o vetor final da consulta?"

    print("\n\n### 1. Baseline: apenas busca densa")
    pipeline.config = Config.basico()
    pipeline.perguntar(pergunta).imprimir()

    print("\n\n### 2. Busca híbrida (denso + BM25 fundidos por RRF)")
    pipeline.config = Config(hibrido=True, ordenacao="bordas")
    pipeline.perguntar(pergunta).imprimir()

    print("\n\n### 3. Pipeline completo (HyDE + reescrita + rerank + crítica)")
    pipeline.config = Config.completo()
    pipeline.perguntar(pergunta, verboso=True).imprimir()

    print("\n\n### 4. Pergunta que não exige recuperação")
    pipeline.perguntar("Escreva um haicai sobre o mar.").imprimir()

    print(
        "\n\nNote na etapa 4: o token Retrieve do Self-RAG detecta que a pergunta "
        "não precisa de busca e pula a recuperação, economizando chamadas e "
        "evitando poluir o prompt com contexto irrelevante."
    )


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BLIS RAG — Módulo 04: RAG Avançado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    def add_corpus(p):
        p.add_argument("--corpus", default=str(DIR_CORPUS))

    def add_topk(p):
        p.add_argument("--top-k", type=int, default=SETTINGS.top_k)

    def add_preset(p):
        p.add_argument(
            "--preset",
            default="completo",
            choices=list(PRESETS),
            help="conjunto de técnicas ativas",
        )

    p = sub.add_parser("indexar", help="indexa o corpus")
    add_corpus(p)
    p.add_argument(
        "--raptor",
        action="store_true",
        help="constrói a árvore de resumos RAPTOR (usa o LLM)",
    )
    p.set_defaults(func=cmd_indexar)

    p = sub.add_parser("perguntar", help="faz uma pergunta")
    p.add_argument("pergunta")
    add_preset(p)
    add_topk(p)
    p.add_argument("--verboso", action="store_true", help="mostra as etapas internas")
    p.add_argument("--sem-contexto", action="store_true")
    p.set_defaults(func=cmd_perguntar)

    p = sub.add_parser("chat", help="modo interativo")
    add_preset(p)
    add_topk(p)
    p.set_defaults(func=cmd_chat)

    p = sub.add_parser("comparar", help="ablação das técnicas")
    add_topk(p)
    p.add_argument(
        "--rapido",
        action="store_true",
        help="só as técnicas que não chamam o LLM (sem custo)",
    )
    p.set_defaults(func=cmd_comparar)

    p = sub.add_parser("bm25", help="inspeciona a busca lexical")
    p.add_argument("pergunta")
    add_topk(p)
    p.set_defaults(func=cmd_bm25)

    p = sub.add_parser("demo", help="roteiro guiado")
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
