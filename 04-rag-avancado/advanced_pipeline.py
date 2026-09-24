"""
Pipeline de RAG avançado.

Encadeia as técnicas dos artigos do módulo 4, cada uma ligável/desligável por
flag, para que o efeito de cada melhoria possa ser medido isoladamente.

Fluxo completo quando tudo está ativo:

    pergunta
      │
      ├─[Retrieve]──── precisa recuperar? (Self-RAG)  ─── não ──> resposta direta
      │ sim
      ├─[reescrita]─── multi-query / step-back
      ├─[HyDE]──────── documento hipotético -> vetor de consulta
      │
      ├─[busca densa]──┐
      │                ├─[RRF]─> candidatos (N=20)
      ├─[busca BM25]───┘
      │
      ├─[IsREL]─────── filtra irrelevantes (Self-RAG)
      ├─[rerank]────── LLM reordena e corta para top-k (k=4)
      ├─[ordenação]─── relevantes nas bordas (Lost in the Middle)
      │
      ├─[geração]───── resposta com citações
      └─[IsSUP/IsUSE]─ auto-crítica e ressalva
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import context_ordering
import query_rewriting
from bm25 import BM25
from chunking import Chunk
from config import SETTINGS, Settings
from embeddings import Embedder, criar_embedder
from fusao import deduplicar, reciprocal_rank_fusion, soma_ponderada
from generator import RespostaGerada, gerar_resposta
from hyde import ResultadoHyDE, vetor_hyde
from ingest import ingerir
from raptor import chunks_da_arvore, construir_arvore
from reranker import filtrar_relevantes, rerank
from self_rag import Critica, anotar_resposta, criticar_resposta, decidir_recuperacao
from vector_store import Resultado, VectorStore

CAMINHO_INDICE = Path("indice/vector_store.json")
CAMINHO_CACHE = Path("indice/cache_embeddings.json")


@dataclass
class Config:
    """Quais técnicas estão ativas nesta execução."""

    hibrido: bool = True
    fusao: str = "rrf"  # "rrf" ou "ponderada"
    hyde: bool = False
    reescrita: str = ""  # "", "multi_query", "step_back", "decompor"
    reranking: str = ""  # "", "pointwise", "listwise"
    filtro_isrel: bool = False
    ordenacao: str = "bordas"  # "bordas", "decrescente", "crescente", "documento"
    retrieve_sob_demanda: bool = False
    autocritica: bool = False

    @classmethod
    def basico(cls) -> "Config":
        """Equivalente ao módulo 03: só busca densa, sem nenhuma melhoria."""
        return cls(hibrido=False, ordenacao="decrescente")

    @classmethod
    def completo(cls) -> "Config":
        """Todas as técnicas ligadas."""
        return cls(
            hibrido=True,
            fusao="rrf",
            hyde=True,
            reescrita="multi_query",
            reranking="pointwise",
            filtro_isrel=True,
            ordenacao="bordas",
            retrieve_sob_demanda=True,
            autocritica=True,
        )

    def ativas(self) -> list[str]:
        nomes = []
        if self.hibrido:
            nomes.append(f"híbrido({self.fusao})")
        if self.hyde:
            nomes.append("HyDE")
        if self.reescrita:
            nomes.append(f"reescrita({self.reescrita})")
        if self.reranking:
            nomes.append(f"rerank({self.reranking})")
        if self.filtro_isrel:
            nomes.append("IsREL")
        if self.retrieve_sob_demanda:
            nomes.append("Retrieve")
        if self.autocritica:
            nomes.append("IsSUP/IsUSE")
        nomes.append(f"ordem({self.ordenacao})")
        return nomes or ["nenhuma"]


@dataclass
class RespostaAvancada:
    """Resultado completo, com o rastro de cada etapa para auditoria."""

    pergunta: str
    resposta: RespostaGerada
    recuperados: list[Resultado] = field(default_factory=list)
    candidatos_brutos: int = 0
    descartados_isrel: int = 0
    consultas_usadas: list[str] = field(default_factory=list)
    hyde: ResultadoHyDE | None = None
    critica: Critica | None = None
    recuperou: bool = True
    motivo_sem_recuperacao: str = ""
    tempos: dict = field(default_factory=dict)

    @property
    def tempo_total(self) -> float:
        return sum(self.tempos.values())

    def imprimir(self, mostrar_contexto: bool = True) -> None:
        print(f"\nPergunta: {self.pergunta}")
        print("=" * 70)

        if not self.recuperou:
            print(f"\n[Retrieve=no] {self.motivo_sem_recuperacao}")
            print(f"\nResposta:\n{self.resposta.texto}")
            return

        if len(self.consultas_usadas) > 1:
            print(f"\nConsultas usadas ({len(self.consultas_usadas)}):")
            for c in self.consultas_usadas:
                print(f"  - {c}")

        if self.hyde and self.hyde.documentos_hipoteticos:
            previa = self.hyde.documentos_hipoteticos[0][:150].replace("\n", " ")
            print(f"\nHyDE (doc. hipotético): {previa}...")

        if mostrar_contexto:
            print(
                f"\nContexto final ({len(self.recuperados)} de "
                f"{self.candidatos_brutos} candidatos"
                + (
                    f", {self.descartados_isrel} descartados por IsREL"
                    if self.descartados_isrel
                    else ""
                )
                + ") — na ordem em que entram no prompt:"
            )
            for r in self.recuperados:
                previa = r.chunk.texto[:100].replace("\n", " ")
                print(f"  {r.rank}. [{r.chunk.chunk_id}] score={r.score:.4f}")
                print(f"     {previa}...")

        print(f"\nResposta:\n{self.resposta.texto}")

        if self.critica:
            self.critica.imprimir()

        tempos = " | ".join(f"{k} {v:.2f}s" for k, v in self.tempos.items())
        print(f"\n({tempos} | total {self.tempo_total:.2f}s)")


class AdvancedRAGPipeline:
    """Pipeline de RAG avançado, com as técnicas configuráveis por flag."""

    def __init__(
        self,
        settings: Settings | None = None,
        config: Config | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or SETTINGS
        self.config = config or Config()
        self.embedder = embedder or criar_embedder(self.settings, cache=CAMINHO_CACHE)
        self.store = store or VectorStore(nome_modelo=self.embedder.nome)
        self.bm25: BM25 | None = None

    # ---------------------------- indexação ----------------------------- #

    def indexar_diretorio(
        self,
        diretorio: str | Path,
        usar_raptor: bool = False,
        verboso: bool = True,
    ) -> int:
        if verboso:
            print("\n[1/3] Ingestão e chunking")
        chunks = ingerir(
            diretorio,
            estrategia="recursivo",
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
            verboso=verboso,
        )

        if usar_raptor:
            if verboso:
                print("\n[1b] Construindo a árvore RAPTOR")
            nos = construir_arvore(
                chunks,
                self.embedder,
                niveis=self.settings.raptor_niveis,
                tamanho_cluster=self.settings.raptor_tamanho_cluster,
                settings=self.settings,
                verboso=verboso,
            )
            chunks = chunks_da_arvore(nos)

        return self.indexar_chunks(chunks, verboso=verboso)

    def indexar_chunks(self, chunks: list[Chunk], verboso: bool = True) -> int:
        if not chunks:
            return 0

        if verboso:
            print(f"\n[2/3] Embeddings ({self.embedder.nome})")
        inicio = time.perf_counter()
        vetores = self.embedder.embed([c.texto for c in chunks])
        if verboso:
            print(
                f"  {len(chunks)} vetores em {time.perf_counter() - inicio:.2f}s "
                f"| dimensão={len(vetores[0])}"
            )

        self.store.nome_modelo = self.embedder.nome
        self.store.adicionar(chunks, vetores)

        if verboso:
            print("\n[3/3] Índice lexical BM25")
        self.bm25 = BM25(self.store.chunks)
        if verboso:
            print(f"  {self.bm25.resumo()}")
            print(f"\nÍndice pronto: {self.store.resumo()}")
        return len(chunks)

    def _garantir_bm25(self) -> BM25:
        if self.bm25 is None:
            self.bm25 = BM25(self.store.chunks)
        return self.bm25

    # ----------------------------- etapas ------------------------------- #

    def _vetores_de_consulta(
        self, consultas: list[str]
    ) -> tuple[list[list[float]], ResultadoHyDE | None]:
        """Constrói o(s) vetor(es) de busca, com ou sem HyDE."""
        if not self.config.hyde:
            return self.embedder.embed(consultas), None

        # HyDE na consulta principal; as variações seguem como embedding direto.
        resultado_hyde = vetor_hyde(
            consultas[0],
            self.embedder,
            n=self.settings.hyde_n,
            settings=self.settings,
        )
        vetores = [resultado_hyde.vetor]
        if len(consultas) > 1:
            vetores.extend(self.embedder.embed(consultas[1:]))
        return vetores, resultado_hyde

    def recuperar(
        self, pergunta: str, top_k: int | None = None
    ) -> list[Resultado]:
        """
        Etapa de recuperação isolada, sem geração.

        É esta função que a avaliação usa para comparar configurações.
        """
        if len(self.store) == 0:
            raise RuntimeError("Índice vazio. Rode indexar_diretorio() antes.")

        top_k = top_k or self.settings.top_k
        n_candidatos = max(self.settings.candidatos, top_k)

        # 1. Reescrita de consultas
        consultas = [pergunta]
        if self.config.reescrita:
            expandidas = query_rewriting.expandir(
                pergunta, self.config.reescrita, self.settings
            )
            consultas = expandidas.todas

        # 2. Busca densa (uma por consulta), com HyDE opcional
        vetores, _ = self._vetores_de_consulta(consultas)
        rankings: list[list[Resultado]] = [
            self.store.buscar(v, top_k=n_candidatos) for v in vetores
        ]

        # 3. Busca lexical BM25 (uma por consulta)
        if self.config.hibrido:
            bm25 = self._garantir_bm25()
            rankings.extend(bm25.buscar(c, top_k=n_candidatos) for c in consultas)

        # 4. Fusão
        if len(rankings) == 1:
            candidatos = rankings[0][:n_candidatos]
        elif self.config.fusao == "ponderada" and len(rankings) == 2:
            candidatos = soma_ponderada(
                rankings[0],
                rankings[1],
                peso_denso=self.settings.peso_denso,
                top_k=n_candidatos,
            )
        else:
            candidatos = reciprocal_rank_fusion(
                rankings, k=self.settings.rrf_k, top_k=n_candidatos
            )

        candidatos = deduplicar(candidatos)

        # 5. Filtro IsREL e reranking
        if self.config.filtro_isrel:
            candidatos, _ = filtrar_relevantes(pergunta, candidatos, self.settings)

        if self.config.reranking:
            finais = rerank(
                pergunta, candidatos, top_k, self.config.reranking, self.settings
            )
        else:
            finais = candidatos[:top_k]

        # 6. Reordenação física do contexto
        return context_ordering.reordenar(finais, self.config.ordenacao)

    # ---------------------------- consulta ------------------------------ #

    def perguntar(
        self, pergunta: str, top_k: int | None = None, verboso: bool = False
    ) -> RespostaAvancada:
        """Pipeline completo, com todas as etapas ativas na configuração."""
        top_k = top_k or self.settings.top_k
        tempos: dict[str, float] = {}

        # Etapa 0: decidir se vale recuperar (token Retrieve do Self-RAG)
        if self.config.retrieve_sob_demanda:
            inicio = time.perf_counter()
            decisao = decidir_recuperacao(pergunta, self.settings)
            tempos["retrieve?"] = time.perf_counter() - inicio

            if not decisao.recuperar:
                from generator import gerar_sem_contexto

                inicio = time.perf_counter()
                resposta = gerar_sem_contexto(pergunta, self.settings)
                tempos["geração"] = time.perf_counter() - inicio
                return RespostaAvancada(
                    pergunta=pergunta,
                    resposta=resposta,
                    recuperou=False,
                    motivo_sem_recuperacao=decisao.motivo,
                    tempos=tempos,
                )

        # Etapas 1-3: reescrita, HyDE, busca, fusão
        inicio = time.perf_counter()
        consultas = [pergunta]
        if self.config.reescrita:
            expandidas = query_rewriting.expandir(
                pergunta, self.config.reescrita, self.settings
            )
            consultas = expandidas.todas
            if verboso:
                expandidas.imprimir()
        tempos["reescrita"] = time.perf_counter() - inicio

        inicio = time.perf_counter()
        vetores, resultado_hyde = self._vetores_de_consulta(consultas)
        if verboso and resultado_hyde:
            resultado_hyde.imprimir()
        tempos["hyde"] = time.perf_counter() - inicio

        inicio = time.perf_counter()
        n_candidatos = max(self.settings.candidatos, top_k)
        rankings = [self.store.buscar(v, top_k=n_candidatos) for v in vetores]

        if self.config.hibrido:
            bm25 = self._garantir_bm25()
            rankings.extend(bm25.buscar(c, top_k=n_candidatos) for c in consultas)

        if len(rankings) == 1:
            candidatos = rankings[0][:n_candidatos]
        elif self.config.fusao == "ponderada" and len(rankings) == 2:
            candidatos = soma_ponderada(
                rankings[0], rankings[1], self.settings.peso_denso, n_candidatos
            )
        else:
            candidatos = reciprocal_rank_fusion(
                rankings, self.settings.rrf_k, n_candidatos
            )
        candidatos = deduplicar(candidatos)
        n_brutos = len(candidatos)
        tempos["busca"] = time.perf_counter() - inicio

        # Etapa 4: filtro IsREL
        descartados = 0
        if self.config.filtro_isrel:
            inicio = time.perf_counter()
            candidatos, fora = filtrar_relevantes(pergunta, candidatos, self.settings)
            descartados = len(fora)
            tempos["IsREL"] = time.perf_counter() - inicio

        # Etapa 5: reranking
        if self.config.reranking:
            inicio = time.perf_counter()
            finais = rerank(
                pergunta, candidatos, top_k, self.config.reranking, self.settings
            )
            tempos["rerank"] = time.perf_counter() - inicio
        else:
            finais = candidatos[:top_k]

        # Etapa 6: ordenação física contra o Lost in the Middle
        antes = list(finais)
        finais = context_ordering.reordenar(finais, self.config.ordenacao)
        if verboso:
            context_ordering.explicar_ordenacao(antes, finais)

        # Etapa 7: geração
        inicio = time.perf_counter()
        resposta = gerar_resposta(pergunta, finais, self.settings)
        tempos["geração"] = time.perf_counter() - inicio

        # Etapa 8: auto-crítica (IsSUP / IsUSE)
        critica = None
        if self.config.autocritica and not resposta.abstencao:
            inicio = time.perf_counter()
            critica = criticar_resposta(
                pergunta, resposta.texto, finais, self.settings
            )
            resposta.texto = anotar_resposta(resposta.texto, critica)
            tempos["crítica"] = time.perf_counter() - inicio

        return RespostaAvancada(
            pergunta=pergunta,
            resposta=resposta,
            recuperados=finais,
            candidatos_brutos=n_brutos,
            descartados_isrel=descartados,
            consultas_usadas=consultas,
            hyde=resultado_hyde,
            critica=critica,
            tempos=tempos,
        )

    # --------------------------- persistência --------------------------- #

    def salvar(self, caminho: str | Path = CAMINHO_INDICE) -> None:
        self.store.salvar(caminho)
        print(f"Índice salvo em {caminho} ({self.store.resumo()})")

    @classmethod
    def carregar(
        cls,
        caminho: str | Path = CAMINHO_INDICE,
        settings: Settings | None = None,
        config: Config | None = None,
    ) -> "AdvancedRAGPipeline":
        settings = settings or SETTINGS
        store = VectorStore.carregar(caminho)
        embedder = criar_embedder(settings, cache=CAMINHO_CACHE)

        if store.nome_modelo and store.nome_modelo != embedder.nome:
            raise RuntimeError(
                f"O índice foi criado com '{store.nome_modelo}', mas o embedder "
                f"atual é '{embedder.nome}'. Reindexe o corpus."
            )

        pipeline = cls(settings=settings, config=config, embedder=embedder, store=store)
        pipeline._garantir_bm25()
        return pipeline
