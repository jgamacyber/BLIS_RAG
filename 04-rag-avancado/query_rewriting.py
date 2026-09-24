"""
Reescrita e expansão de consultas.

A consulta do usuário raramente está na forma ideal para recuperação: é curta,
ambígua, ou usa um vocabulário diferente do corpus. Três técnicas:

- multi_query: gera N reformulações e busca com todas, fundindo os resultados.
  Ataca a lacuna de vocabulário por cobertura.
- step_back: gera uma pergunta mais geral, que recupera o contexto conceitual
  de fundo, complementando o detalhe específico da pergunta original.
- decompor: quebra perguntas compostas em subperguntas independentes, para
  perguntas que exigem múltiplos saltos de raciocínio.

Todas custam ao menos uma chamada extra ao LLM antes da busca. Em produção, vale
medir se o ganho de revocação compensa a latência.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from config import SETTINGS, Settings, get_client

PROMPT_MULTI_QUERY = """Você reescreve consultas de busca para melhorar a recuperação \
de documentos.

Gere {n} reformulações diferentes da pergunta abaixo. Cada reformulação deve:
- preservar a intenção original;
- usar vocabulário ou enfoque diferente das outras (sinônimos, termos técnicos, \
formulação mais direta);
- ser uma pergunta ou frase de busca autocontida.

Responda APENAS com as {n} reformulações, uma por linha, sem numeração e sem \
comentários.

Pergunta: {pergunta}"""

PROMPT_STEP_BACK = """Dada a pergunta específica abaixo, formule uma pergunta mais \
geral e abstrata, sobre o conceito ou o mecanismo de fundo que seria necessário \
entender para responder à original.

Exemplo:
Pergunta específica: "Qual o valor de k1 usado no BM25 neste artigo?"
Pergunta geral: "Como funciona a fórmula do BM25 e quais são seus parâmetros?"

Responda APENAS com a pergunta geral, sem preâmbulo.

Pergunta específica: {pergunta}"""

PROMPT_DECOMPOR = """Decomponha a pergunta abaixo em subperguntas simples e \
independentes, que juntas permitam respondê-la.

Se a pergunta já for simples e não exigir decomposição, responda apenas com ela \
mesma, inalterada.

Responda APENAS com as subperguntas, uma por linha, sem numeração.

Pergunta: {pergunta}"""


@dataclass
class ConsultasExpandidas:
    """Resultado de uma expansão, sempre incluindo a consulta original."""

    original: str
    variacoes: list[str] = field(default_factory=list)
    tecnica: str = ""

    @property
    def todas(self) -> list[str]:
        """Consulta original em primeiro lugar, seguida das variações."""
        vistas = {self.original.strip().lower()}
        saida = [self.original]
        for v in self.variacoes:
            chave = v.strip().lower()
            if chave and chave not in vistas:
                vistas.add(chave)
                saida.append(v)
        return saida

    def imprimir(self) -> None:
        print(f"\nExpansão ({self.tecnica}) — {len(self.todas)} consultas:")
        for i, consulta in enumerate(self.todas):
            marca = "original" if i == 0 else f"var {i}"
            print(f"  [{marca:>8}] {consulta}")


def _chamar(prompt: str, settings: Settings, max_tokens: int = 300) -> str:
    client = get_client(settings)
    resposta = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        # Temperatura baixa mas não zero: queremos variação controlada.
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return (resposta.choices[0].message.content or "").strip()


def _linhas(texto: str) -> list[str]:
    """Extrai linhas limpas, removendo numeração e marcadores do LLM."""
    linhas = []
    for linha in texto.splitlines():
        linha = linha.strip()
        if not linha:
            continue
        # Remove "1.", "1)", "-", "*" no início.
        linha = re.sub(r"^\s*(\d+[.)]|[-*•])\s*", "", linha).strip()
        # Remove aspas envolventes.
        linha = linha.strip('"').strip("'").strip()
        if linha:
            linhas.append(linha)
    return linhas


def multi_query(
    pergunta: str, n: int = 3, settings: Settings | None = None
) -> ConsultasExpandidas:
    """Gera N reformulações da pergunta com vocabulários diferentes."""
    settings = settings or SETTINGS
    try:
        texto = _chamar(PROMPT_MULTI_QUERY.format(n=n, pergunta=pergunta), settings)
        variacoes = _linhas(texto)[:n]
    except Exception as erro:  # noqa: BLE001
        print(f"  [multi-query] falha ({erro}); seguindo só com a consulta original")
        variacoes = []
    return ConsultasExpandidas(pergunta, variacoes, tecnica="multi-query")


def step_back(pergunta: str, settings: Settings | None = None) -> ConsultasExpandidas:
    """Gera uma pergunta mais geral, para recuperar o contexto conceitual."""
    settings = settings or SETTINGS
    try:
        texto = _chamar(
            PROMPT_STEP_BACK.format(pergunta=pergunta), settings, max_tokens=120
        )
        variacoes = _linhas(texto)[:1]
    except Exception as erro:  # noqa: BLE001
        print(f"  [step-back] falha ({erro}); seguindo só com a consulta original")
        variacoes = []
    return ConsultasExpandidas(pergunta, variacoes, tecnica="step-back")


def decompor(
    pergunta: str, settings: Settings | None = None
) -> ConsultasExpandidas:
    """Quebra uma pergunta composta em subperguntas independentes."""
    settings = settings or SETTINGS
    try:
        texto = _chamar(PROMPT_DECOMPOR.format(pergunta=pergunta), settings)
        variacoes = _linhas(texto)[:4]
    except Exception as erro:  # noqa: BLE001
        print(f"  [decompor] falha ({erro}); seguindo só com a consulta original")
        variacoes = []
    return ConsultasExpandidas(pergunta, variacoes, tecnica="decomposição")


TECNICAS = {
    "multi_query": multi_query,
    "step_back": step_back,
    "decompor": decompor,
}


def expandir(
    pergunta: str,
    tecnica: str = "multi_query",
    settings: Settings | None = None,
    **kwargs,
) -> ConsultasExpandidas:
    """Despacha para a técnica escolhida."""
    if tecnica not in TECNICAS:
        raise ValueError(
            f"Técnica '{tecnica}' desconhecida. Disponíveis: {', '.join(TECNICAS)}"
        )
    if tecnica == "multi_query":
        return multi_query(
            pergunta, n=kwargs.get("n", (settings or SETTINGS).n_reescritas),
            settings=settings,
        )
    return TECNICAS[tecnica](pergunta, settings=settings)
