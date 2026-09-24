"""
Ingestão de documentos: carregar arquivos do disco e transformá-los em chunks.

Suporta .txt, .md e .json. PDF é opcional: se `pypdf` estiver instalado, também
é lido; caso contrário o arquivo é ignorado com um aviso, em vez de quebrar o
pipeline inteiro.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chunking import Chunk, dividir_documento, estatisticas

EXTENSOES_TEXTO = {".txt", ".md", ".markdown"}


@dataclass
class Documento:
    """Um documento bruto, antes do chunking."""

    doc_id: str
    titulo: str
    texto: str
    origem: str = ""

    def __len__(self) -> int:
        return len(self.texto)


# --------------------------------------------------------------------------- #
# Leitores
# --------------------------------------------------------------------------- #


def _titulo_do_markdown(texto: str, padrao: str) -> str:
    """Usa o primeiro '# título' do arquivo, se houver."""
    for linha in texto.splitlines():
        linha = linha.strip()
        if linha.startswith("# "):
            return linha[2:].strip()
        if linha:
            break
    return padrao


def ler_texto(caminho: Path) -> Documento:
    texto = caminho.read_text(encoding="utf-8", errors="replace")
    return Documento(
        doc_id=caminho.stem,
        titulo=_titulo_do_markdown(texto, caminho.stem),
        texto=texto,
        origem=str(caminho),
    )


def ler_pdf(caminho: Path) -> Documento | None:
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"  [aviso] pypdf não instalado; ignorando {caminho.name}")
        return None

    try:
        leitor = PdfReader(str(caminho))
        paginas = [(p.extract_text() or "") for p in leitor.pages]
    except Exception as erro:  # noqa: BLE001
        print(f"  [aviso] falha ao ler {caminho.name}: {erro}")
        return None

    texto = "\n\n".join(p.strip() for p in paginas if p.strip())
    if not texto.strip():
        print(f"  [aviso] {caminho.name} não tem texto extraível (PDF escaneado?)")
        return None

    return Documento(
        doc_id=caminho.stem,
        titulo=caminho.stem,
        texto=texto,
        origem=str(caminho),
    )


def carregar_documentos(diretorio: str | Path) -> list[Documento]:
    """Varre um diretório recursivamente e devolve os documentos legíveis."""
    diretorio = Path(diretorio)
    if not diretorio.exists():
        raise FileNotFoundError(f"Diretório não encontrado: {diretorio}")

    documentos: list[Documento] = []
    for caminho in sorted(diretorio.rglob("*")):
        if not caminho.is_file():
            continue

        sufixo = caminho.suffix.lower()
        if sufixo in EXTENSOES_TEXTO:
            documentos.append(ler_texto(caminho))
        elif sufixo == ".pdf":
            doc = ler_pdf(caminho)
            if doc:
                documentos.append(doc)

    if not documentos:
        raise ValueError(
            f"Nenhum documento legível em {diretorio} "
            f"(esperado: {', '.join(sorted(EXTENSOES_TEXTO))} ou .pdf)"
        )
    return documentos


# --------------------------------------------------------------------------- #
# Pipeline de ingestão
# --------------------------------------------------------------------------- #


def ingerir(
    diretorio: str | Path,
    estrategia: str = "recursivo",
    chunk_size: int = 600,
    overlap: int = 100,
    verboso: bool = True,
) -> list[Chunk]:
    """Carrega os documentos de um diretório e devolve a lista de chunks."""
    documentos = carregar_documentos(diretorio)
    if verboso:
        print(f"  {len(documentos)} documento(s) carregado(s) de {diretorio}")

    chunks: list[Chunk] = []
    for doc in documentos:
        novos = dividir_documento(
            texto=doc.texto,
            doc_id=doc.doc_id,
            titulo=doc.titulo,
            estrategia=estrategia,
            chunk_size=chunk_size,
            overlap=overlap,
            metadados={"origem": doc.origem},
        )
        chunks.extend(novos)
        if verboso:
            print(f"    - {doc.doc_id}: {len(doc):>6} chars -> {len(novos):>3} chunks")

    if verboso:
        st = estatisticas(chunks)
        print(
            f"  Total: {st['n_chunks']} chunks | "
            f"média {st['media']} chars (min {st['min']}, max {st['max']})"
        )
    return chunks
