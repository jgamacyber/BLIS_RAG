"""
HyDE — Hypothetical Document Embeddings.

Implementação de Gao, Ma, Lin & Callan (2022), "Precise Zero-Shot Dense
Retrieval without Relevance Labels".

A ideia: em vez de embeddar a pergunta e procurar documentos parecidos com uma
*pergunta*, pede-se ao LLM que escreva um documento hipotético que responda à
pergunta, e embedda-se esse documento. A busca passa a ser
documento-contra-documento, que é o regime em que os codificadores contrastivos
foram treinados.

O documento gerado não precisa ser verdadeiro. Ele pode conter erros factuais e
detalhes inventados — espera-se apenas que capture o *padrão de relevância*. O
codificador funciona como um compressor com perdas: ao projetar o texto em um
vetor denso, os detalhes alucinados são filtrados, e o vetor resultante fica
ancorado na vizinhança correta do corpus real.

Agregação (Eq. 8 do paper):

    v_q = (1 / (N+1)) · [ Σ f(d̂_k) + f(q) ]
                          k=1..N

Ou seja: média dos N documentos hipotéticos codificados MAIS a consulta
codificada. Incluir a consulta original é uma salvaguarda — se a geração sair
péssima, a busca não é completamente desviada.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import SETTINGS, Settings, get_client
from embeddings import Embedder

# O paper usa instruções específicas por tarefa. Esta é a forma geral,
# equivalente ao "write a paragraph that answers the question".
INSTRUCAO_PADRAO = """Escreva um parágrafo que responda à pergunta abaixo.

Escreva como se fosse um trecho de um documento técnico ou artigo científico \
sobre o assunto. Seja específico e use a terminologia da área. Não diga que não \
sabe e não peça mais informação: escreva a melhor resposta plausível que \
conseguir, mesmo que não tenha certeza dos detalhes.

Responda apenas com o parágrafo, sem preâmbulo.

Pergunta: {pergunta}"""

INSTRUCOES = {
    "padrao": INSTRUCAO_PADRAO,
    "cientifico": (
        "Escreva um trecho de artigo científico que responda à pergunta. "
        "Use linguagem acadêmica e mencione métodos, métricas ou resultados "
        "quando fizer sentido. Responda apenas com o trecho.\n\n"
        "Pergunta: {pergunta}"
    ),
    "web": (
        "Escreva um parágrafo de uma página web que responda à pergunta, "
        "em linguagem direta. Responda apenas com o parágrafo.\n\n"
        "Pergunta: {pergunta}"
    ),
}


@dataclass
class ResultadoHyDE:
    """Saída do HyDE, com os documentos gerados para auditoria."""

    pergunta: str
    documentos_hipoteticos: list[str] = field(default_factory=list)
    vetor: list[float] = field(default_factory=list)
    incluiu_consulta: bool = True

    def imprimir(self, largura: int = 300) -> None:
        print(f"\nHyDE — {len(self.documentos_hipoteticos)} documento(s) hipotético(s):")
        for i, doc in enumerate(self.documentos_hipoteticos, start=1):
            previa = doc[:largura].replace("\n", " ")
            print(f"  [{i}] {previa}{'...' if len(doc) > largura else ''}")


def gerar_documentos_hipoteticos(
    pergunta: str,
    n: int = 1,
    instrucao: str = "padrao",
    settings: Settings | None = None,
) -> list[str]:
    """Amostra N documentos hipotéticos do LLM para a pergunta."""
    settings = settings or SETTINGS
    client = get_client(settings)
    template = INSTRUCOES.get(instrucao, INSTRUCAO_PADRAO)

    documentos: list[str] = []
    for _ in range(max(1, n)):
        resposta = client.chat.completions.create(
            model=settings.chat_model,
            messages=[{"role": "user", "content": template.format(pergunta=pergunta)}],
            # Temperatura 0.7, como no paper: gerações diversas entre si.
            temperature=settings.hyde_temperatura,
            max_tokens=300,
        )
        texto = (resposta.choices[0].message.content or "").strip()
        if texto:
            documentos.append(texto)

    return documentos


def _media_vetores(vetores: list[list[float]]) -> list[float]:
    """Média elemento a elemento de uma lista de vetores."""
    if not vetores:
        return []
    n = len(vetores)
    dim = len(vetores[0])
    return [sum(v[i] for v in vetores) / n for i in range(dim)]


def vetor_hyde(
    pergunta: str,
    embedder: Embedder,
    n: int = 1,
    incluir_consulta: bool = True,
    instrucao: str = "padrao",
    settings: Settings | None = None,
) -> ResultadoHyDE:
    """
    Constrói o vetor de consulta do HyDE (Eq. 8 do paper).

    Se a geração falhar (rede, cota, filtro), cai de volta no embedding da
    consulta original em vez de quebrar o pipeline.
    """
    settings = settings or SETTINGS

    try:
        documentos = gerar_documentos_hipoteticos(pergunta, n, instrucao, settings)
    except Exception as erro:  # noqa: BLE001
        print(f"  [hyde] falha ao gerar hipóteses ({erro}); usando a consulta original")
        documentos = []

    textos = list(documentos)
    if incluir_consulta or not documentos:
        textos.append(pergunta)

    vetores = embedder.embed(textos)
    return ResultadoHyDE(
        pergunta=pergunta,
        documentos_hipoteticos=documentos,
        vetor=_media_vetores(vetores),
        incluiu_consulta=incluir_consulta or not documentos,
    )
