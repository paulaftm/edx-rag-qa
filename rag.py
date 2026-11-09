"""Pipeline RAG local: PDF -> chunks -> embeddings -> ChromaDB-> resposta com fontes.

- LLM: Ollama (padrão: IBM Granite 4).
- Embeddings: ibm-granite/granite-embedding-278m-multilingual via Hugging Face.
- Base vetorial: Chroma, persistida em disco, uma coleção por PDF.
"""

import hashlib
import os
import time
from functools import lru_cache
from pathlib import Path
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------
# Configuração das variáveis de ambiente
# --------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "granite4:micro")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "ibm-granite/granite-embedding-278m-multilingual"
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("TOP_K", "4"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "350"))
INDEX_DIR = Path(os.getenv("INDEX_DIR", ".chroma"))

# Prompts de configuração do chatbot (em pt-br):
PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Você é um assistente que responde perguntas sobre um documento. "
            "Use APENAS os trechos fornecidos no contexto. "
            "Cite os trechos que sustentam a resposta pelo rótulo, ex.: (C1). "
            "Não use colchetes para citar: o documento pode ter suas próprias notas "
            "numeradas entre colchetes, que não têm relação com os trechos do contexto. "
            "Se o contexto não contiver a resposta, diga claramente que não encontrou "
            "a informação no documento, sem inventar. "
            "Responda no mesmo idioma da pergunta.",
        ),
        ("human", "Contexto:\n{context}\n\nPergunta: {question}"),
    ]
)


# ------------------------------------------------------------------------------
# Componentes (funções reutilizáveis com cache "Least Recently Used" definido):
# ------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Modelo de embedding local; baixado do Hugging Face na primeira execução."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """LLM local servido pelo Ollama."""
    return ChatOllama(model=LLM_MODEL, temperature=TEMPERATURE, num_predict=MAX_TOKENS)


def file_hash(path: str) -> str:
    """Identifica o PDF pelo conteúdo, para não reindexar o mesmo arquivo."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]



# Indexação
def load_pdf(path: str) -> list[Document]:
    return PyPDFLoader(path).load()


def split(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


@lru_cache(maxsize=16)
def get_vectorstore(pdf_path: str, digest: str) -> Chroma:
    """Uma coleção por PDF (pelo hash): nada se mistura entre documentos.

    Se o índice já existe em disco, é reaproveitado; senão, é criado.
    """
    persist_dir = INDEX_DIR / digest
    collection = f"pdf_{digest}"
    if persist_dir.exists():
        return Chroma(
            collection_name=collection,
            embedding_function=get_embeddings(),
            persist_directory=str(persist_dir),
        )
    chunks = split(load_pdf(pdf_path))
    if not chunks:
        raise ValueError("Não foi possível extrair texto do PDF.")
    return Chroma.from_documents(
        chunks,
        get_embeddings(),
        collection_name=collection,
        persist_directory=str(persist_dir),
    )


# -------------------
# Pergunta e resposta
# -------------------
def format_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page") # metadados
        label = f"p. {page + 1}" if isinstance(page, int) else "página desconhecida"
        parts.append(f"C{i} ({label}):\n{doc.page_content}") # texto do trecho
    return "\n\n".join(parts)


def answer(pdf_path: str, question: str) -> tuple[str, str, float, float, float]:
    """Responde à pergunta.

    Devolve (resposta, trechos usados, tempo de recuperação, tempo de
    geração, tempo total) - tempos em segundos.
    """
    if not pdf_path:
        return "Envie um PDF primeiro.", "", 0.0, 0.0, 0.0
    if not question or not question.strip():
        return "Digite uma pergunta.", "", 0.0, 0.0, 0.0

    t0 = time.perf_counter()
    store = get_vectorstore(pdf_path, file_hash(pdf_path))
    docs = store.as_retriever(search_kwargs={"k": TOP_K}).invoke(question)
    context = format_context(docs)
    t1 = time.perf_counter()

    messages = PROMPT.invoke({"context": context, "question": question})
    response = get_llm().invoke(messages)
    t2 = time.perf_counter()

    retrieval_time = round(t1 - t0, 1)
    generation_time = round(t2 - t1, 1)
    total_time = round(t2 - t0, 1)
    return response.content, context, retrieval_time, generation_time, total_time
