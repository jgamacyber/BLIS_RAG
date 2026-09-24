"""
Testes com LLM simulado — validam os caminhos que chamam a API, sem gastar nada.

Os testes offline (`testes_offline.py`) cobrem o que não depende do LLM. Este
arquivo cobre o resto: substitui o cliente da OpenRouter por um dublê que
devolve respostas controladas, e verifica que:

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
import hyde as mod_hyde
import query_rewriting as mod_rewriting
import reranker as mod_reranker
import self_rag as mod_selfrag
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
        mod_hyde,
        mod_rewriting,
        mod_reranker,
        mod_selfrag,
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

    candidatos = 6
    peso_denso = 0.5
    rrf_k = 60
    hyde_n = 2
    hyde_temperatura = 0.7
    n_reescritas = 3
    raptor_niveis = 2
    raptor_tamanho_cluster = 5


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


def testar_hyde() -> None:
    secao("3. HyDE")
    from embeddings import LocalHashingEmbedder

    instalar_cliente(["Documento hipotético A.", "Documento hipotético B."])
    embedder = LocalHashingEmbedder(dimensao=64)

    resultado = mod_hyde.vetor_hyde("pergunta?", embedder, n=2, settings=S)

    checar(len(resultado.documentos_hipoteticos) == 2, "gera N documentos hipotéticos")
    checar(len(CHAMADAS) == 2, "faz uma chamada por documento hipotético")
    checar(
        CHAMADAS[0]["temperature"] == 0.7,
        "usa temperatura 0.7 nas hipóteses (valor do artigo)",
    )
    checar(resultado.incluiu_consulta, "inclui a consulta original na agregação (Eq. 8)")
    checar(len(resultado.vetor) == 64, "o vetor final tem a dimensão do embedder")

    # A média de N+1 vetores deve diferir do embedding só da consulta.
    so_consulta = embedder.embed_um("pergunta?")
    checar(
        resultado.vetor != so_consulta,
        "o vetor HyDE difere do embedding puro da consulta",
    )

    # Fallback quando a geração quebra
    class ClienteQuebrado:
        def __init__(self):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kw: (_ for _ in ()).throw(RuntimeError("falha"))
                )
            )

    mod_hyde.get_client = lambda settings=None: ClienteQuebrado()
    fallback = mod_hyde.vetor_hyde("pergunta?", embedder, n=1, settings=S)
    checar(fallback.documentos_hipoteticos == [], "falha na geração não quebra")
    checar(
        fallback.vetor == so_consulta,
        "com falha, cai de volta no embedding da consulta",
    )


def testar_reescrita() -> None:
    secao("4. Reescrita de consultas")
    instalar_cliente(["1. Primeira variação\n2. Segunda variação\n- Terceira variação"])

    expandidas = mod_rewriting.multi_query("pergunta original?", n=3, settings=S)

    checar(len(expandidas.variacoes) == 3, "extrai as 3 variações")
    checar(
        expandidas.variacoes[0] == "Primeira variação",
        "remove a numeração '1.' das linhas",
    )
    checar(
        expandidas.variacoes[2] == "Terceira variação",
        "remove o marcador '-' das linhas",
    )
    checar(expandidas.todas[0] == "pergunta original?", "a original vem em primeiro")
    checar(len(expandidas.todas) == 4, "total = original + 3 variações")

    # Deduplicação
    instalar_cliente(["pergunta original?\nOutra forma"])
    dedup = mod_rewriting.multi_query("pergunta original?", n=2, settings=S)
    checar(len(dedup.todas) == 2, "não duplica uma variação igual à original")

    instalar_cliente(["Como funciona o mecanismo em geral?"])
    sb = mod_rewriting.step_back("Qual o valor exato do parâmetro k1?", settings=S)
    checar(len(sb.variacoes) == 1, "step-back devolve uma única pergunta geral")


def testar_reranking() -> None:
    secao("5. Reranking")
    instalar_cliente(["3", "9", "7"])

    reordenados = mod_reranker.rerank_pointwise("pergunta?", _resultados(3), top_k=3, settings=S)

    checar(len(reordenados) == 3, "devolve o top_k pedido")
    checar(len(CHAMADAS) == 3, "faz uma chamada por candidato (pointwise)")
    checar(
        reordenados[0].chunk.chunk_id == "doc2#002",
        "o candidato com nota 9 sobe para o primeiro lugar",
    )
    checar(
        reordenados[-1].chunk.chunk_id == "doc1#001",
        "o candidato com nota 3 cai para o último",
    )
    checar(
        all(0.0 <= r.score <= 1.0 for r in reordenados),
        "as notas são normalizadas para [0,1]",
    )
    checar(
        all(r.rank == i + 1 for i, r in enumerate(reordenados)),
        "ranks renumerados após o reranking",
    )

    # Resposta malformada não deve eliminar o candidato
    instalar_cliente(["não sei dizer", "8", "2"])
    robusto = mod_reranker.rerank_pointwise("p?", _resultados(3), top_k=3, settings=S)
    checar(len(robusto) == 3, "nota ilegível não descarta o candidato")

    # Listwise
    instalar_cliente(["3,1,2"])
    lista = mod_reranker.rerank_listwise("p?", _resultados(3), top_k=3, settings=S)
    checar(len(CHAMADAS) == 1, "listwise faz uma única chamada")
    checar(
        [r.chunk.chunk_id for r in lista]
        == ["doc3#003", "doc1#001", "doc2#002"],
        "listwise aplica a ordem devolvida pelo modelo",
    )

    instalar_cliente(["2"])  # modelo esqueceu dois itens
    incompleto = mod_reranker.rerank_listwise("p?", _resultados(3), top_k=3, settings=S)
    checar(len(incompleto) == 3, "listwise completa os itens que o modelo omitiu")

    # IsREL
    instalar_cliente(['{"relevante": true}', '{"relevante": false}', '{"relevante": true}'])
    relevantes, fora = mod_reranker.filtrar_relevantes("p?", _resultados(3), S)
    checar(len(relevantes) == 2 and len(fora) == 1, "IsREL separa relevantes de irrelevantes")

    instalar_cliente(['{"relevante": false}'] * 3)
    todos, _ = mod_reranker.filtrar_relevantes("p?", _resultados(3), S)
    checar(len(todos) == 3, "se IsREL descarta tudo, mantém todos (não zera o contexto)")


def testar_self_rag() -> None:
    secao("6. Self-RAG (tokens de reflexão)")

    instalar_cliente(['{"recuperar": false, "motivo": "pedido criativo"}'])
    decisao = mod_selfrag.decidir_recuperacao("Escreva um poema.", S)
    checar(not decisao.recuperar, "Retrieve=no para pedido criativo")
    checar(decisao.motivo == "pedido criativo", "captura o motivo da decisão")

    instalar_cliente(['{"recuperar": true, "motivo": "fato específico"}'])
    checar(
        mod_selfrag.decidir_recuperacao("Quantos parâmetros tem o modelo?", S).recuperar,
        "Retrieve=yes para pergunta factual",
    )

    instalar_cliente(["resposta sem json nenhum"])
    checar(
        mod_selfrag.decidir_recuperacao("pergunta?", S).recuperar,
        "resposta ilegível assume recuperar (padrão seguro)",
    )

    instalar_cliente(
        ['{"suporte": "fully", "utilidade": 5, "afirmacoes_sem_respaldo": [], '
         '"comentario": "ok"}']
    )
    critica = mod_selfrag.criticar_resposta("p?", "resposta", _resultados(), S)
    checar(critica.suporte == "fully" and critica.utilidade == 5, "lê IsSUP e IsUSE")
    checar(critica.confiavel, "fully + 5 é considerado confiável")
    checar(
        mod_selfrag.anotar_resposta("texto", critica) == "texto",
        "resposta confiável não recebe ressalva",
    )

    instalar_cliente(
        ['{"suporte": "partially", "utilidade": 3, '
         '"afirmacoes_sem_respaldo": ["o custo de treino"], "comentario": "parcial"}']
    )
    parcial = mod_selfrag.criticar_resposta("p?", "resposta", _resultados(), S)
    checar(not parcial.confiavel, "partially não é considerado confiável")
    anotada = mod_selfrag.anotar_resposta("texto", parcial)
    checar("Ressalva" in anotada, "resposta não confiável recebe ressalva")
    checar("o custo de treino" in anotada, "a ressalva lista as afirmações sem respaldo")

    instalar_cliente(['{"suporte": "inventado", "utilidade": 99}'])
    saneada = mod_selfrag.criticar_resposta("p?", "r", _resultados(), S)
    checar(saneada.suporte == "partially", "valor inválido de IsSUP cai no padrão")
    checar(saneada.utilidade == 5, "IsUSE fora da escala é limitado a 1..5")

    vazia = mod_selfrag.criticar_resposta("p?", "r", [], S)
    checar(vazia.suporte == "no_support", "sem contexto, a crítica é no_support")


def testar_pipeline_completo() -> None:
    secao("7. Pipeline avançado ponta a ponta (tudo simulado)")
    import advanced_pipeline as mod_pipeline
    from embeddings import LocalHashingEmbedder
    from advanced_pipeline import AdvancedRAGPipeline, Config
    from vector_store import VectorStore

    chunks = [
        Chunk(
            texto=f"Conteúdo {i} sobre recuperação densa, BM25 e reranking de passagens.",
            doc_id=f"doc{i}",
            chunk_id=f"doc{i}#000",
            titulo=f"Doc {i}",
        )
        for i in range(1, 7)
    ]
    embedder = LocalHashingEmbedder(dimensao=64)
    store = VectorStore(nome_modelo=embedder.nome)
    store.adicionar(chunks, embedder.embed([c.texto for c in chunks]))

    pipeline = AdvancedRAGPipeline(
        settings=S, config=Config.completo(), embedder=embedder, store=store
    )

    respostas = [
        '{"recuperar": true, "motivo": "factual"}',   # Retrieve
        "variação um\nvariação dois",                  # multi-query
        "documento hipotético gerado",                 # HyDE
    ] + ['{"relevante": true}'] * 6 + ["8"] * 6 + [     # IsREL e reranking
        "Resposta final [doc1#000].",                  # geração
        '{"suporte": "fully", "utilidade": 5, "afirmacoes_sem_respaldo": []}',
    ]
    instalar_cliente(respostas)
    mod_pipeline.filtrar_relevantes = mod_reranker.filtrar_relevantes

    resultado = pipeline.perguntar("O que é reranking?", top_k=3)

    checar(resultado.recuperou, "o pipeline decidiu recuperar")
    checar(len(resultado.recuperados) == 3, "devolve o top_k pedido")
    checar(resultado.candidatos_brutos >= 3, "registra o número de candidatos brutos")
    checar(len(resultado.consultas_usadas) == 3, "usou a original + 2 variações")
    checar(resultado.hyde is not None, "o HyDE foi aplicado")
    checar(resultado.critica is not None, "a auto-crítica foi executada")
    checar(bool(resultado.tempos), "registra o tempo de cada etapa")
    checar("Resposta final" in resultado.resposta.texto, "a resposta final é produzida")

    # Retrieve = no: deve pular toda a recuperação
    instalar_cliente(
        ['{"recuperar": false, "motivo": "criativo"}', "Um haicai qualquer."]
    )
    criativa = pipeline.perguntar("Escreva um haicai.", top_k=3)
    checar(not criativa.recuperou, "Retrieve=no pula a recuperação")
    checar(criativa.recuperados == [], "sem recuperação, não há passagens")
    checar(len(CHAMADAS) == 2, "Retrieve=no economiza chamadas (só decisão + geração)")


def main() -> int:
    print("=" * 62)
    print("  TESTES COM LLM SIMULADO — sem API, sem custo")
    print("=" * 62)

    testar_embeddings()
    testar_geracao()
    testar_hyde()
    testar_reescrita()
    testar_reranking()
    testar_self_rag()
    testar_pipeline_completo()

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
