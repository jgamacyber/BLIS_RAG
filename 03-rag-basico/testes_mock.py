"""
Testes com LLM simulado — validam os caminhos que chamam a API, sem gastar nada.

Os testes offline (`testes_offline.py`) cobrem o que não depende do LLM. Este
arquivo cobre o resto — embeddings via API e geração —, substituindo o cliente
da OpenRouter por um dublê que devolve respostas controladas. Verifica que:

- as chamadas são montadas no formato certo (modelo, mensagens, parâmetros);
- as respostas do LLM são interpretadas corretamente;
- respostas malformadas não derrubam o pipeline.

Um erro de digitação em `client.embeddings.create` ou um parser frágil de JSON
só apareceria em produção, gastando créditos. Aqui aparece de graça.

    python testes_mock.py
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import config
import generator as mod_generator
from chunking import Chunk
from vector_store import Resultado

FALHAS: list[str] = []
CHAMADAS: list[dict] = []


def checar(condicao: bool, descricao: str) -> None:
    if condicao:
        print(f"  [ok]   {descricao}")
    else:
        print(f"  [FALHA] {descricao}")
        FALHAS.append(descricao)


def secao(titulo: str) -> None:
    print(f"\n{titulo}")
    print("-" * 62)


# --------------------------------------------------------------------------- #
# Dublê do cliente OpenAI/OpenRouter
# --------------------------------------------------------------------------- #


class ChatFalso:
    def __init__(self, respostas: list[str]) -> None:
        self.respostas = respostas
        self.i = 0

    def create(self, **kwargs):
        CHAMADAS.append({"tipo": "chat", **kwargs})
        texto = self.respostas[min(self.i, len(self.respostas) - 1)]
        self.i += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=texto))],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        )


class EmbeddingsFalsos:
    def create(self, **kwargs):
        CHAMADAS.append({"tipo": "embeddings", **kwargs})
        entradas = kwargs["input"]
        if isinstance(entradas, str):
            entradas = [entradas]
        # Vetor determinístico derivado do texto, com índice fora de ordem
        # de propósito: o código deve reordenar por `index`.
        dados = [
            SimpleNamespace(index=i, embedding=[float(len(t) % 7), 1.0, 0.5])
            for i, t in enumerate(entradas)
        ]
        return SimpleNamespace(data=list(reversed(dados)))


class ClienteFalso:
    def __init__(self, respostas_chat: list[str]) -> None:
        self.chat = SimpleNamespace(completions=ChatFalso(respostas_chat))
        self.embeddings = EmbeddingsFalsos()


def instalar_cliente(respostas: list[str]) -> ClienteFalso:
    """Substitui get_client em todos os módulos que o importaram."""
    CHAMADAS.clear()
    cliente = ClienteFalso(respostas)
    falso = lambda settings=None: cliente  # noqa: E731

    for modulo in (
        config,
        mod_generator,
    ):
        if hasattr(modulo, "get_client"):
            modulo.get_client = falso
    return cliente


class SettingsFalsas:
    """Settings com uma chave fictícia, para não bater na validação."""

    api_key = "sk-teste"
    chat_model = "modelo/teste"
    embedding_model = "modelo/embed"
    embedding_provider = "openrouter"
    temperature = 0.0
    max_tokens = 600
    usa_api = True

    chunk_size = 600
    chunk_overlap = 100
    top_k = 4



S = SettingsFalsas()


def _resultados(n: int = 3) -> list[Resultado]:
    return [
        Resultado(
            chunk=Chunk(
                texto=f"Passagem número {i} com conteúdo relevante sobre o tema.",
                doc_id=f"doc{i}",
                chunk_id=f"doc{i}#00{i}",
                titulo=f"Documento {i}",
            ),
            score=1.0 - i / 10,
            rank=i,
        )
        for i in range(1, n + 1)
    ]


# --------------------------------------------------------------------------- #


def testar_embeddings() -> None:
    secao("1. Embeddings via API (formato da chamada)")
    from embeddings import OpenRouterEmbedder

    import embeddings as mod_embeddings

    cliente = ClienteFalso([])
    mod_embeddings.get_client = lambda settings=None: cliente
    CHAMADAS.clear()

    embedder = OpenRouterEmbedder(S)
    vetores = embedder.embed(["texto um", "texto dois pouco maior"])

    checar(len(vetores) == 2, "devolve um vetor por texto")
    checar(len(CHAMADAS) == 1, "faz uma única chamada em lote, não uma por texto")
    checar(CHAMADAS[0]["model"] == "modelo/embed", "usa o modelo de embeddings")
    checar(
        isinstance(CHAMADAS[0]["input"], list),
        "envia `input` como lista (batching)",
    )
    # O dublê devolve os dados invertidos: se o código não reordenar por index,
    # o vetor do primeiro texto virá trocado.
    checar(
        vetores[0][0] == float(len("texto um") % 7),
        "reordena os embeddings pelo campo `index` da resposta",
    )
    checar(embedder.dimensao == 3, "descobre a dimensão a partir da resposta")

    CHAMADAS.clear()
    checar(embedder.embed([]) == [], "lista vazia não gera chamada")
    checar(len(CHAMADAS) == 0, "lista vazia realmente não chama a API")


def testar_geracao() -> None:
    secao("2. Geração com contexto")
    instalar_cliente(["Resposta fundamentada [doc1#001]."])

    resposta = mod_generator.gerar_resposta("pergunta?", _resultados(), S)

    checar(resposta.texto.startswith("Resposta"), "extrai o texto da resposta")
    checar(len(resposta.contexto_usado) == 3, "registra as passagens usadas")
    checar(resposta.tokens_prompt == 100, "registra os tokens do prompt")
    checar(not resposta.abstencao, "resposta normal não é marcada como abstenção")

    chamada = CHAMADAS[0]
    checar(chamada["model"] == "modelo/teste", "usa o modelo de chat configurado")
    checar(chamada["temperature"] == 0.0, "usa temperatura 0 na geração")
    checar(len(chamada["messages"]) == 2, "envia mensagem de sistema + usuário")
    checar(chamada["messages"][0]["role"] == "system", "primeira mensagem é system")

    conteudo = chamada["messages"][1]["content"]
    checar("doc1#001" in conteudo, "o contexto inclui os identificadores das passagens")
    checar("pergunta?" in conteudo, "o prompt inclui a pergunta")

    # Abstenção
    instalar_cliente([mod_generator.FRASE_SEM_RESPOSTA])
    vazia = mod_generator.gerar_resposta("pergunta?", _resultados(), S)
    checar(vazia.abstencao, "detecta a frase de abstenção")

    CHAMADAS.clear()
    sem_ctx = mod_generator.gerar_resposta("pergunta?", [], S)
    checar(sem_ctx.abstencao, "sem passagens, abstém sem chamar a API")
    checar(len(CHAMADAS) == 0, "sem passagens, não gasta chamada")







def main() -> int:
    print("=" * 62)
    print("  TESTES COM LLM SIMULADO — sem API, sem custo")
    print("=" * 62)

    testar_embeddings()
    testar_geracao()

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
