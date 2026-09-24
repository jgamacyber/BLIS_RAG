"""
Geração de embeddings.

Dois provedores:

1. OpenRouterEmbedder — usa POST /api/v1/embeddings da OpenRouter, compatível
   com o SDK da OpenAI. É o modo real.
2. LocalHashingEmbedder — embedder determinístico baseado em hashing de
   n-gramas, sem rede e sem custo. Serve APENAS para testar o encanamento do
   pipeline offline: a qualidade semântica é baixa (é um saco-de-n-gramas
   projetado, não um modelo treinado).

Contexto dos papers:
- Karpukhin et al. (2020), DPR: bi-encoder BERT, d=768, similaridade por produto
  interno; embeddings densos capturam sinônimos e paráfrases que o BM25 perde.
- Muennighoff et al. (2023), MTEB: 8 tarefas, 58 datasets, 112 idiomas. A
  conclusão prática é que NÃO existe um embedder universalmente melhor — a
  escolha deve ser validada na sua tarefa e no seu idioma.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path

from config import SETTINGS, Settings, get_client


class Embedder(ABC):
    """Interface mínima de um gerador de embeddings."""

    dimensao: int
    nome: str

    @abstractmethod
    def embed(self, textos: list[str]) -> list[list[float]]:
        """Converte uma lista de textos em uma lista de vetores."""

    def embed_um(self, texto: str) -> list[float]:
        return self.embed([texto])[0]


# --------------------------------------------------------------------------- #
# Provedor real: OpenRouter
# --------------------------------------------------------------------------- #


class OpenRouterEmbedder(Embedder):
    """
    Embeddings via OpenRouter.

    Faz batching (a API aceita uma lista em `input`) e repete a chamada com
    backoff exponencial em caso de erro transitório.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        batch_size: int = 64,
        max_tentativas: int = 4,
    ) -> None:
        self.settings = settings or SETTINGS
        self.client = get_client(self.settings)
        self.nome = self.settings.embedding_model
        self.batch_size = batch_size
        self.max_tentativas = max_tentativas
        self.dimensao = 0  # descoberta na primeira chamada

    def embed(self, textos: list[str]) -> list[list[float]]:
        if not textos:
            return []

        vetores: list[list[float]] = []
        for inicio in range(0, len(textos), self.batch_size):
            lote = textos[inicio : inicio + self.batch_size]
            vetores.extend(self._embed_lote(lote))
        if vetores:
            self.dimensao = len(vetores[0])
        return vetores

    def _embed_lote(self, lote: list[str]) -> list[list[float]]:
        # A API rejeita strings vazias; troca por um espaço e segue.
        lote = [t if t.strip() else " " for t in lote]

        ultimo_erro: Exception | None = None
        for tentativa in range(self.max_tentativas):
            try:
                resposta = self.client.embeddings.create(
                    model=self.settings.embedding_model,
                    input=lote,
                )
                # A ordem de `data` pode não ser garantida: ordena por index.
                itens = sorted(resposta.data, key=lambda d: d.index)
                return [list(item.embedding) for item in itens]
            except Exception as erro:  # noqa: BLE001 - queremos repetir em qualquer falha de rede
                ultimo_erro = erro
                if tentativa == self.max_tentativas - 1:
                    break
                espera = 2**tentativa
                print(f"  [embeddings] falha ({erro}); nova tentativa em {espera}s")
                time.sleep(espera)

        raise RuntimeError(
            f"Falha ao gerar embeddings após {self.max_tentativas} tentativas: "
            f"{ultimo_erro}"
        )


# --------------------------------------------------------------------------- #
# Provedor offline: hashing determinístico
# --------------------------------------------------------------------------- #


class LocalHashingEmbedder(Embedder):
    """
    Embedder offline por hashing de n-gramas de palavras, com pesagem TF-IDF-like
    (sublinear tf) e normalização L2.

    NÃO é um modelo semântico: dois textos com vocabulário diferente mas mesmo
    significado ficarão distantes. Use apenas para validar o pipeline sem API.
    """

    def __init__(self, dimensao: int = 512, ngramas: tuple[int, ...] = (1, 2)) -> None:
        self.dimensao = dimensao
        self.ngramas = ngramas
        self.nome = f"local-hashing-{dimensao}d"

    _TOKEN = re.compile(r"[a-zà-ÿ0-9]+", re.IGNORECASE)

    def _tokenizar(self, texto: str) -> list[str]:
        return self._TOKEN.findall(texto.lower())

    def _termos(self, texto: str) -> list[str]:
        tokens = self._tokenizar(texto)
        termos: list[str] = []
        for n in self.ngramas:
            if n == 1:
                termos.extend(tokens)
            else:
                termos.extend(
                    " ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)
                )
        return termos

    def _indice(self, termo: str) -> tuple[int, float]:
        """Posição no vetor e sinal, derivados de um hash estável."""
        digest = hashlib.md5(termo.encode("utf-8")).digest()
        pos = int.from_bytes(digest[:4], "big") % self.dimensao
        sinal = 1.0 if digest[4] % 2 == 0 else -1.0
        return pos, sinal

    def embed(self, textos: list[str]) -> list[list[float]]:
        vetores: list[list[float]] = []
        for texto in textos:
            vetor = [0.0] * self.dimensao
            contagem: dict[str, int] = {}
            for termo in self._termos(texto):
                contagem[termo] = contagem.get(termo, 0) + 1

            for termo, freq in contagem.items():
                pos, sinal = self._indice(termo)
                # tf sublinear: reduz o peso de termos muito repetidos
                vetor[pos] += sinal * (1.0 + math.log(freq))

            norma = math.sqrt(sum(v * v for v in vetor))
            if norma > 0:
                vetor = [v / norma for v in vetor]
            vetores.append(vetor)
        return vetores


# --------------------------------------------------------------------------- #
# Cache em disco
# --------------------------------------------------------------------------- #


class EmbedderComCache(Embedder):
    """
    Decorador que memoriza embeddings em um JSON local.

    Reindexar o mesmo corpus é a operação mais repetida durante o estudo; sem
    cache, cada execução recalcula tudo e consome créditos à toa.
    """

    def __init__(self, base: Embedder, caminho: str | Path) -> None:
        self.base = base
        self.caminho = Path(caminho)
        self.nome = base.nome
        self.dimensao = base.dimensao
        self._cache: dict[str, list[float]] = {}
        self._carregar()

    def _carregar(self) -> None:
        if self.caminho.exists():
            try:
                dados = json.loads(self.caminho.read_text(encoding="utf-8"))
                if dados.get("modelo") == self.base.nome:
                    self._cache = dados.get("vetores", {})
            except (json.JSONDecodeError, OSError):
                self._cache = {}

    def _salvar(self) -> None:
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self.caminho.write_text(
            json.dumps(
                {"modelo": self.base.nome, "vetores": self._cache}, ensure_ascii=False
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _chave(texto: str) -> str:
        return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:32]

    def embed(self, textos: list[str]) -> list[list[float]]:
        chaves = [self._chave(t) for t in textos]
        faltantes = [t for t, k in zip(textos, chaves) if k not in self._cache]

        if faltantes:
            # Deduplica antes de chamar a API.
            unicos = list(dict.fromkeys(faltantes))
            novos = self.base.embed(unicos)
            for texto, vetor in zip(unicos, novos):
                self._cache[self._chave(texto)] = vetor
            self._salvar()

        self.dimensao = self.base.dimensao or (
            len(next(iter(self._cache.values()))) if self._cache else 0
        )
        return [self._cache[k] for k in chaves]


# --------------------------------------------------------------------------- #
# Fábrica
# --------------------------------------------------------------------------- #


def criar_embedder(
    settings: Settings | None = None, cache: str | Path | None = None
) -> Embedder:
    """Instancia o embedder conforme EMBEDDING_PROVIDER, com cache opcional."""
    settings = settings or SETTINGS

    if settings.usa_api:
        base: Embedder = OpenRouterEmbedder(settings)
    else:
        base = LocalHashingEmbedder()
        print(
            "  [aviso] usando embedder LOCAL de hashing (offline). "
            "A qualidade semântica é baixa — use apenas para testar o pipeline."
        )

    if cache:
        return EmbedderComCache(base, cache)
    return base
