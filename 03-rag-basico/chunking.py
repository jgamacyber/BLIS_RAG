"""
Estratégias de chunking (divisão de documentos em passagens).

O chunk é a unidade de recuperação: é ele que será embeddado, indexado e
devolvido como contexto. Chunk grande demais dilui o sinal do embedding e gasta
contexto; pequeno demais perde a informação necessária para responder.

Referências:
- Lewis et al. (2020), RAG: Wikipedia dividida em blocos disjuntos de 100 palavras.
- Karpukhin et al. (2020), DPR: mesma escolha, passagens de 100 palavras.
- Sarthi et al. (2024), RAPTOR: chunks de ~100 tokens que NUNCA cortam uma frase
  no meio — se a frase estoura o limite, ela inteira vai para o próximo chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class Chunk:
    """Uma passagem indexável, com metadados de origem."""

    texto: str
    doc_id: str
    chunk_id: str
    titulo: str = ""
    posicao: int = 0
    metadados: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.texto)

    def para_contexto(self) -> str:
        """Formato usado ao montar o prompt do gerador."""
        cabecalho = self.titulo or self.doc_id
        return f"[{self.chunk_id}] {cabecalho}\n{self.texto}"


# --------------------------------------------------------------------------- #
# Utilidades de segmentação
# --------------------------------------------------------------------------- #

# Divide em frases respeitando abreviações comuns em português.
_ABREVIACOES = r"(?<!\bSr)(?<!\bSra)(?<!\bDr)(?<!\bDra)(?<!\bProf)(?<!\bex)(?<!\bEx)"
_FIM_DE_FRASE = re.compile(rf"{_ABREVIACOES}(?<=[.!?])\s+(?=[A-ZÀ-Ú0-9])")


def dividir_em_frases(texto: str) -> list[str]:
    """Segmenta um texto em frases. Heurística simples, sem dependências."""
    texto = re.sub(r"\s+", " ", texto).strip()
    if not texto:
        return []
    frases = [f.strip() for f in _FIM_DE_FRASE.split(texto) if f.strip()]
    return frases or [texto]


def dividir_em_paragrafos(texto: str) -> list[str]:
    """Segmenta por linha em branco, preservando a estrutura do documento."""
    partes = re.split(r"\n\s*\n", texto)
    return [p.strip() for p in partes if p.strip()]


# --------------------------------------------------------------------------- #
# Estratégias
# --------------------------------------------------------------------------- #


def chunk_por_tamanho_fixo(
    texto: str, chunk_size: int = 600, overlap: int = 100
) -> list[str]:
    """
    Janela deslizante de caracteres. É a estratégia mais ingênua: rápida,
    previsível e capaz de cortar palavras e frases no meio.

    Serve de baseline para comparar com as estratégias melhores.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size deve ser positivo")
    if overlap >= chunk_size:
        raise ValueError("overlap deve ser menor que chunk_size")

    texto = texto.strip()
    if not texto:
        return []

    passo = chunk_size - overlap
    pedacos: list[str] = []
    for inicio in range(0, len(texto), passo):
        pedaco = texto[inicio : inicio + chunk_size].strip()
        if pedaco:
            pedacos.append(pedaco)
        if inicio + chunk_size >= len(texto):
            break
    return pedacos


def chunk_por_frases(
    texto: str, chunk_size: int = 600, overlap: int = 100
) -> list[str]:
    """
    Agrupa frases inteiras até encher o orçamento de caracteres.

    Nunca corta uma frase ao meio (estratégia do RAPTOR). A sobreposição é feita
    reaproveitando as últimas frases do chunk anterior, o que preserva o contexto
    na fronteira entre chunks.
    """
    frases = dividir_em_frases(texto)
    if not frases:
        return []

    pedacos: list[str] = []
    atual: list[str] = []
    tamanho = 0

    for frase in frases:
        # Frase isolada maior que o limite: vira um chunk próprio, sem cortar.
        if len(frase) > chunk_size and not atual:
            pedacos.append(frase)
            continue

        if tamanho + len(frase) + 1 > chunk_size and atual:
            pedacos.append(" ".join(atual))
            # Reaproveita o final do chunk anterior como sobreposição.
            atual, tamanho = _cauda_para_overlap(atual, overlap)

        atual.append(frase)
        tamanho += len(frase) + 1

    if atual:
        pedacos.append(" ".join(atual))
    return pedacos


def _cauda_para_overlap(frases: list[str], overlap: int) -> tuple[list[str], int]:
    """Devolve as últimas frases que cabem no orçamento de sobreposição."""
    if overlap <= 0:
        return [], 0
    cauda: list[str] = []
    total = 0
    for frase in reversed(frases):
        if total + len(frase) > overlap:
            break
        cauda.insert(0, frase)
        total += len(frase) + 1
    return cauda, total


def chunk_recursivo(
    texto: str, chunk_size: int = 600, overlap: int = 100
) -> list[str]:
    """
    Estratégia hierárquica: tenta respeitar parágrafos; se um parágrafo não cabe,
    desce para frases; se uma frase ainda não cabe, corta por tamanho fixo.

    É o padrão recomendado deste módulo — preserva a estrutura do documento na
    maior parte dos casos e degrada de forma controlada nos casos difíceis.
    """
    paragrafos = dividir_em_paragrafos(texto)
    if not paragrafos:
        return []

    pedacos: list[str] = []
    buffer: list[str] = []
    tamanho = 0

    def descarregar() -> None:
        nonlocal buffer, tamanho
        if buffer:
            pedacos.append("\n\n".join(buffer))
            buffer, tamanho = [], 0

    for paragrafo in paragrafos:
        if len(paragrafo) > chunk_size:
            # Parágrafo grande: fecha o buffer e quebra o parágrafo por frases.
            descarregar()
            pedacos.extend(chunk_por_frases(paragrafo, chunk_size, overlap))
            continue

        if tamanho + len(paragrafo) + 2 > chunk_size and buffer:
            descarregar()

        buffer.append(paragrafo)
        tamanho += len(paragrafo) + 2

    descarregar()
    return pedacos


ESTRATEGIAS = {
    "fixo": chunk_por_tamanho_fixo,
    "frases": chunk_por_frases,
    "recursivo": chunk_recursivo,
}


def dividir_documento(
    texto: str,
    doc_id: str,
    titulo: str = "",
    estrategia: str = "recursivo",
    chunk_size: int = 600,
    overlap: int = 100,
    metadados: dict | None = None,
) -> list[Chunk]:
    """Aplica uma estratégia e devolve objetos Chunk já identificados."""
    if estrategia not in ESTRATEGIAS:
        raise ValueError(
            f"Estratégia '{estrategia}' desconhecida. "
            f"Disponíveis: {', '.join(ESTRATEGIAS)}"
        )

    textos = ESTRATEGIAS[estrategia](texto, chunk_size, overlap)
    return [
        Chunk(
            texto=t,
            doc_id=doc_id,
            chunk_id=f"{doc_id}#{i:03d}",
            titulo=titulo,
            posicao=i,
            metadados=dict(metadados or {}, estrategia=estrategia),
        )
        for i, t in enumerate(textos)
    ]


def estatisticas(chunks: Iterable[Chunk]) -> dict:
    """Métricas descritivas para comparar estratégias de chunking."""
    tamanhos = [len(c) for c in chunks]
    if not tamanhos:
        return {"n_chunks": 0, "media": 0, "min": 0, "max": 0, "total": 0}
    return {
        "n_chunks": len(tamanhos),
        "media": round(sum(tamanhos) / len(tamanhos), 1),
        "min": min(tamanhos),
        "max": max(tamanhos),
        "total": sum(tamanhos),
    }
