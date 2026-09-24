"""
Orquestração do pipeline de RAG básico.

Indexação (offline):
    documentos -> chunking -> embeddings -> vector store

Consulta (online):
    pergunta -> embedding -> busca top-k -> prompt com contexto -> resposta
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from chunking import Chunk
from config import SETTINGS, Settings
from embeddings import Embedder, criar_embedder
from generator import RespostaGerada, gerar_resposta
from ingest import ingerir
from vector_store import Resultado, VectorStore

CAMINHO_INDICE = Path("indice/vector_store.json")
CAMINHO_CACHE = Path("indice/cache_embeddings.json")


@dataclass
class RespostaRAG:
    """Resultado completo de uma consulta, com rastro para auditoria."""

    pergunta: str
    resposta: RespostaGerada
    recuperados: list[Resultado] = field(default_factory=list)
    tempo_busca: float = 0.0
    tempo_geracao: float = 0.0

    @property
    def tempo_total(self) -> float:
        return self.tempo_busca + self.tempo_geracao

    def imprimir(self, mostrar_contexto: bool = True) -> None:
        print(f"\nPergunta: {self.pergunta}")
        print("=" * 70)

        if mostrar_contexto:
            print(f"\nPassagens recuperadas ({len(self.recuperados)}):")
            for r in self.recuperados:
                previa = r.chunk.texto[:110].replace("\n", " ")
                print(f"  {r.rank}. [{r.chunk.chunk_id}] score={r.score:.4f}")
                print(f"     {previa}...")

        print(f"\nResposta:\n{self.resposta.texto}")
        print(
            f"\n(busca {self.tempo_busca:.2f}s | geração {self.tempo_geracao:.2f}s "
            f"| tokens {self.resposta.tokens_prompt}+{self.resposta.tokens_resposta})"
        )


class RAGPipeline:
    """Pipeline de RAG básico: indexa um corpus e responde perguntas sobre ele."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
    ) -> None:
        self.settings = settings or SETTINGS
        self.embedder = embedder or criar_embedder(self.settings, cache=CAMINHO_CACHE)
        self.store = store or VectorStore(nome_modelo=self.embedder.nome)

    # ---------------------------- indexação ----------------------------- #

    def indexar_diretorio(
        self,
        diretorio: str | Path,
        estrategia: str = "recursivo",
        verboso: bool = True,
    ) -> int:
        """Ingere, divide e indexa todos os documentos de um diretório."""
        if verboso:
            print("\n[1/3] Ingestão e chunking")
        chunks = ingerir(
            diretorio,
            estrategia=estrategia,
            chunk_size=self.settings.chunk_size,
            overlap=self.settings.chunk_overlap,
            verboso=verboso,
        )
        return self.indexar_chunks(chunks, verboso=verboso)

    def indexar_chunks(self, chunks: list[Chunk], verboso: bool = True) -> int:
        """Gera embeddings para os chunks e os adiciona ao índice."""
        if not chunks:
            return 0

        if verboso:
            print(f"\n[2/3] Embeddings ({self.embedder.nome})")
            print(f"  gerando vetores para {len(chunks)} chunks...")

        inicio = time.perf_counter()
        vetores = self.embedder.embed([c.texto for c in chunks])
        duracao = time.perf_counter() - inicio

        self.store.nome_modelo = self.embedder.nome
        self.store.adicionar(chunks, vetores)

        if verboso:
            print(f"  concluído em {duracao:.2f}s | dimensão={len(vetores[0])}")
            print(f"\n[3/3] Índice pronto: {self.store.resumo()}")
        return len(chunks)

    # ----------------------------- consulta ----------------------------- #

    def recuperar(self, pergunta: str, top_k: int | None = None) -> list[Resultado]:
        """Etapa de recuperação isolada (útil para avaliar o retriever sozinho)."""
        if len(self.store) == 0:
            raise RuntimeError("Índice vazio. Rode indexar_diretorio() antes.")
        top_k = top_k or self.settings.top_k
        vetor = self.embedder.embed_um(pergunta)
        return self.store.buscar(vetor, top_k=top_k)

    def perguntar(self, pergunta: str, top_k: int | None = None) -> RespostaRAG:
        """Pipeline completo: recupera o contexto e gera a resposta."""
        inicio = time.perf_counter()
        recuperados = self.recuperar(pergunta, top_k=top_k)
        tempo_busca = time.perf_counter() - inicio

        inicio = time.perf_counter()
        resposta = gerar_resposta(pergunta, recuperados, self.settings)
        tempo_geracao = time.perf_counter() - inicio

        return RespostaRAG(
            pergunta=pergunta,
            resposta=resposta,
            recuperados=recuperados,
            tempo_busca=tempo_busca,
            tempo_geracao=tempo_geracao,
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
    ) -> "RAGPipeline":
        settings = settings or SETTINGS
        store = VectorStore.carregar(caminho)
        embedder = criar_embedder(settings, cache=CAMINHO_CACHE)

        if store.nome_modelo and store.nome_modelo != embedder.nome:
            raise RuntimeError(
                f"O índice foi criado com '{store.nome_modelo}', mas o embedder "
                f"atual é '{embedder.nome}'. Vetores de modelos diferentes não "
                f"são comparáveis — reindexe o corpus."
            )
        return cls(settings=settings, embedder=embedder, store=store)
