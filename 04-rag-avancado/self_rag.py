"""
Self-RAG por prompting — reflexão sobre recuperação e geração.

Implementação inspirada em Asai et al. (2023), "Self-RAG: Learning to Retrieve,
Generate, and Critique through Self-Reflection".

IMPORTANTE — diferença em relação ao paper: o Self-RAG original TREINA um modelo
para emitir tokens de reflexão como parte do vocabulário, usando dados
anotados por um modelo crítico. Aqui não há treinamento: os mesmos sinais são
obtidos por prompting de um LLM de instrução via API. O comportamento é análogo,
a mecânica é diferente, e os resultados não são comparáveis aos do paper.

Os quatro tokens de reflexão implementados:

    Retrieve  {yes, no}                         precisa recuperar?
    IsREL     {relevant, irrelevant}            a passagem é útil?
    IsSUP     {fully, partially, no support}    a resposta é sustentada?
    IsUSE     {1..5}                            a resposta é útil?

O ganho prático é duplo: evita recuperação desnecessária (economia e menos
ruído) e detecta respostas não sustentadas pelo contexto antes de entregá-las.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from config import SETTINGS, Settings, get_client
from vector_store import Resultado


# --------------------------------------------------------------------------- #
# Utilidade: extrair JSON de uma resposta de LLM
# --------------------------------------------------------------------------- #


def _extrair_json(texto: str) -> dict:
    """Extrai o primeiro objeto JSON da resposta. Devolve {} se não achar."""
    match = re.search(r"\{.*\}", texto or "", re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _chamar(prompt: str, settings: Settings, max_tokens: int = 200) -> str:
    client = get_client(settings)
    resposta = client.chat.completions.create(
        model=settings.chat_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    return (resposta.choices[0].message.content or "").strip()


# --------------------------------------------------------------------------- #
# Retrieve — decide se vale a pena recuperar
# --------------------------------------------------------------------------- #

PROMPT_RETRIEVE = """Decida se responder à pergunta abaixo exige buscar informação \
em uma base de documentos externa.

Exige recuperação: perguntas sobre fatos específicos, dados, definições técnicas, \
conteúdo de documentos, números, nomes, datas.
NÃO exige recuperação: pedidos criativos, redação livre, aritmética simples, \
tradução, reformulação de texto já fornecido, conversa social.

Pergunta: {pergunta}

Responda APENAS com um JSON:
{{"recuperar": true/false, "motivo": "uma frase curta"}}"""


@dataclass
class DecisaoRecuperacao:
    recuperar: bool
    motivo: str = ""


def decidir_recuperacao(
    pergunta: str, settings: Settings | None = None
) -> DecisaoRecuperacao:
    """Token Retrieve: recuperar sob demanda em vez de sempre."""
    settings = settings or SETTINGS
    try:
        dados = _extrair_json(
            _chamar(PROMPT_RETRIEVE.format(pergunta=pergunta), settings, 100)
        )
    except Exception as erro:  # noqa: BLE001
        print(f"  [Retrieve] falha ({erro}); assumindo que deve recuperar")
        return DecisaoRecuperacao(True, "falha na decisão; padrão seguro")

    if "recuperar" not in dados:
        # Na dúvida, recupera: falso negativo aqui custa uma resposta sem fonte.
        return DecisaoRecuperacao(True, "resposta ambígua; padrão seguro")
    return DecisaoRecuperacao(
        bool(dados["recuperar"]), str(dados.get("motivo", ""))
    )


# --------------------------------------------------------------------------- #
# IsSUP / IsUSE — crítica da resposta gerada
# --------------------------------------------------------------------------- #

PROMPT_CRITICA = """Avalie a resposta abaixo em relação ao contexto que a originou.

CONTEXTO:
{contexto}

PERGUNTA: {pergunta}

RESPOSTA: {resposta}

Avalie dois aspectos:

1. suporte (IsSUP): as afirmações da resposta são sustentadas pelo contexto?
   - "fully": todas as afirmações são sustentadas pelo contexto
   - "partially": parte é sustentada, parte não tem respaldo
   - "no_support": a resposta não é sustentada pelo contexto

2. utilidade (IsUSE): a resposta é útil para quem perguntou, numa escala de 1 a 5
   (5 = responde completamente; 1 = não responde nada)

Responda APENAS com um JSON:
{{"suporte": "fully|partially|no_support", "utilidade": 1-5, \
"afirmacoes_sem_respaldo": ["..."], "comentario": "uma frase"}}"""


@dataclass
class Critica:
    """Resultado da auto-reflexão sobre uma resposta gerada."""

    suporte: str = "partially"
    utilidade: int = 3
    afirmacoes_sem_respaldo: list[str] = field(default_factory=list)
    comentario: str = ""

    @property
    def confiavel(self) -> bool:
        """Critério para entregar a resposta sem ressalva."""
        return self.suporte == "fully" and self.utilidade >= 4

    @property
    def rotulo(self) -> str:
        mapa = {
            "fully": "totalmente sustentada",
            "partially": "parcialmente sustentada",
            "no_support": "sem respaldo no contexto",
        }
        return mapa.get(self.suporte, self.suporte)

    def imprimir(self) -> None:
        print("\n  Auto-reflexão (Self-RAG):")
        print(f"    IsSUP  {self.suporte:<12} ({self.rotulo})")
        print(f"    IsUSE  {self.utilidade}/5")
        if self.afirmacoes_sem_respaldo:
            print("    Afirmações sem respaldo no contexto:")
            for a in self.afirmacoes_sem_respaldo:
                print(f"      - {a}")
        if self.comentario:
            print(f"    Comentário: {self.comentario}")


def criticar_resposta(
    pergunta: str,
    resposta: str,
    resultados: list[Resultado],
    settings: Settings | None = None,
) -> Critica:
    """Tokens IsSUP e IsUSE: verifica suporte factual e utilidade."""
    settings = settings or SETTINGS
    if not resultados:
        return Critica(
            suporte="no_support", utilidade=1, comentario="nenhum contexto recuperado"
        )

    contexto = "\n\n".join(r.chunk.para_contexto() for r in resultados)

    try:
        dados = _extrair_json(
            _chamar(
                PROMPT_CRITICA.format(
                    contexto=contexto[:8000], pergunta=pergunta, resposta=resposta
                ),
                settings,
                400,
            )
        )
    except Exception as erro:  # noqa: BLE001
        print(f"  [crítica] falha ({erro})")
        return Critica(comentario="falha ao avaliar")

    if not dados:
        return Critica(comentario="resposta da crítica não interpretável")

    utilidade = dados.get("utilidade", 3)
    try:
        utilidade = max(1, min(5, int(utilidade)))
    except (TypeError, ValueError):
        utilidade = 3

    suporte = str(dados.get("suporte", "partially")).lower()
    if suporte not in {"fully", "partially", "no_support"}:
        suporte = "partially"

    afirmacoes = dados.get("afirmacoes_sem_respaldo", [])
    if not isinstance(afirmacoes, list):
        afirmacoes = []

    return Critica(
        suporte=suporte,
        utilidade=utilidade,
        afirmacoes_sem_respaldo=[str(a) for a in afirmacoes][:5],
        comentario=str(dados.get("comentario", "")),
    )


def anotar_resposta(texto: str, critica: Critica) -> str:
    """
    Acrescenta uma ressalva à resposta quando a crítica não a considera confiável.

    É o comportamento honesto: em vez de esconder a incerteza, o sistema a expõe
    ao usuário, que decide se confia.
    """
    if critica.confiavel:
        return texto

    aviso = [
        "",
        "---",
        f"Ressalva da auto-avaliação: resposta {critica.rotulo} "
        f"(utilidade {critica.utilidade}/5).",
    ]
    if critica.afirmacoes_sem_respaldo:
        aviso.append("Afirmações sem respaldo direto no contexto recuperado:")
        aviso.extend(f"  - {a}" for a in critica.afirmacoes_sem_respaldo)
    return texto + "\n".join(aviso)
