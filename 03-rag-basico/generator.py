"""
Geração da resposta condicionada ao contexto recuperado.

É a etapa "G" do RAG. O que importa aqui é o contrato do prompt: o modelo deve
responder SOMENTE com base nas passagens fornecidas, citar de onde tirou cada
afirmação e admitir quando o contexto não contém a resposta.

Referência: Lewis et al. (2020) concatenam a consulta com a passagem recuperada
e deixam o gerador (BART) produzir a resposta, marginalizando sobre as top-K
passagens. Com LLMs de instrução via API não há marginalização — em vez disso
montamos um único prompt com as K passagens e pedimos atribuição explícita.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import SETTINGS, Settings, get_client
from vector_store import Resultado

PROMPT_SISTEMA = """Você é um assistente de pesquisa que responde EXCLUSIVAMENTE \
com base nas passagens fornecidas.

Regras:
1. Use apenas informação presente nas passagens. Não recorra a conhecimento próprio.
2. Cite a passagem que sustenta cada afirmação usando o identificador entre \
colchetes, por exemplo [rag_lewis#002].
3. Se as passagens não contiverem a resposta, diga exatamente: \
"O contexto fornecido não contém informação suficiente para responder." \
Não tente adivinhar.
4. Se as passagens se contradisserem, aponte a contradição em vez de escolher um lado.
5. Responda em português, de forma direta e objetiva."""

TEMPLATE_USUARIO = """Passagens recuperadas:

{contexto}

---
Pergunta: {pergunta}

Responda seguindo as regras, citando os identificadores das passagens usadas."""

FRASE_SEM_RESPOSTA = (
    "O contexto fornecido não contém informação suficiente para responder."
)


@dataclass
class RespostaGerada:
    """Saída do gerador, com rastro do que foi usado."""

    texto: str
    pergunta: str
    contexto_usado: list[str] = field(default_factory=list)
    modelo: str = ""
    tokens_prompt: int = 0
    tokens_resposta: int = 0

    @property
    def abstencao(self) -> bool:
        """True se o modelo declarou que o contexto era insuficiente."""
        return FRASE_SEM_RESPOSTA.lower() in self.texto.lower()


def montar_contexto(resultados: list[Resultado], separador: str = "\n\n") -> str:
    """Concatena as passagens recuperadas no formato que o prompt espera."""
    return separador.join(r.chunk.para_contexto() for r in resultados)


def gerar_resposta(
    pergunta: str,
    resultados: list[Resultado],
    settings: Settings | None = None,
    prompt_sistema: str = PROMPT_SISTEMA,
) -> RespostaGerada:
    """Chama o LLM com a pergunta e as passagens recuperadas."""
    settings = settings or SETTINGS

    if not resultados:
        return RespostaGerada(
            texto=FRASE_SEM_RESPOSTA,
            pergunta=pergunta,
            modelo=settings.chat_model,
        )

    contexto = montar_contexto(resultados)
    client = get_client(settings)

    resposta = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {"role": "system", "content": prompt_sistema},
            {
                "role": "user",
                "content": TEMPLATE_USUARIO.format(
                    contexto=contexto, pergunta=pergunta
                ),
            },
        ],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    uso = getattr(resposta, "usage", None)
    return RespostaGerada(
        texto=(resposta.choices[0].message.content or "").strip(),
        pergunta=pergunta,
        contexto_usado=[r.chunk.chunk_id for r in resultados],
        modelo=settings.chat_model,
        tokens_prompt=getattr(uso, "prompt_tokens", 0) or 0,
        tokens_resposta=getattr(uso, "completion_tokens", 0) or 0,
    )


def gerar_sem_contexto(
    pergunta: str, settings: Settings | None = None
) -> RespostaGerada:
    """
    Baseline "closed-book": responde sem nenhuma recuperação.

    É a comparação que dá sentido ao RAG. Lewis et al. (2020) e Liu et al. (2023)
    usam exatamente esse baseline para medir o ganho real da recuperação.
    """
    settings = settings or SETTINGS
    client = get_client(settings)

    resposta = client.chat.completions.create(
        model=settings.chat_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "Responda à pergunta em português com base apenas no seu "
                    "conhecimento. Se não souber, diga que não sabe."
                ),
            },
            {"role": "user", "content": pergunta},
        ],
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    return RespostaGerada(
        texto=(resposta.choices[0].message.content or "").strip(),
        pergunta=pergunta,
        modelo=settings.chat_model,
    )
