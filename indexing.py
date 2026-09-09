"""Carregamento e indexação de documentos em vectorstore para agentes especialistas.
Autor: Rodrigo Aguiar
Data: 09/09/2026
"""

import os
from typing import Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from loguru import logger

from config import (
    DOCS_PATH,
    FAISS_INDEX_PATH,
    EMBEDDING_MODEL,
    COHERE_API_KEY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SPECIALIST_DOCUMENTS,
    SPECIALIST_FAISS_INDEXES,
)

def load_documents(file_path: str):
    """Carrega um documento PDF específico a partir do caminho fornecido."""
    logger.info(f"Carregando documento: {file_path}")
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        logger.info(f"Documento '{file_path}' carregado. Total de páginas: {len(documents)}")
        return documents
    except Exception as e:
        logger.error(f"Erro ao carregar o documento '{file_path}': {e}")
        raise

def split_documents(documents):
    """Divide os documentos carregados em chunks menores para indexação."""
    logger.info(f"Dividindo documentos em chunks (chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP})...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Total de chunks criados: {len(chunks)}")
    return chunks

def get_embeddings() -> CohereEmbeddings:
    """Inicializa e retorna o modelo de embeddings do Cohere."""
    logger.info(f"Inicializando modelo de embeddings: {EMBEDDING_MODEL}")
    return CohereEmbeddings(model=EMBEDDING_MODEL, cohere_api_key=COHERE_API_KEY)

def create_and_save_faiss_index(chunks, index_path: str, embeddings: CohereEmbeddings) -> Optional[FAISS]:
    """Cria um novo índice FAISS a partir dos chunks e o salva localmente."""
    if not chunks:
        logger.warning(f"Nenhum chunk fornecido para criar o índice em {index_path}. Pulando.")
        return None

    logger.info(f"Criando índice FAISS em: {index_path}")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(index_path)
    logger.success(f"Índice FAISS salvo em: {index_path}")
    return vector_store

def load_faiss_index(index_path: str, embeddings: CohereEmbeddings) -> FAISS:
    """Carrega um índice FAISS existente de um caminho local."""
    logger.info(f"Carregando índice FAISS de: {index_path}")
    try:
        return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        logger.error(f"Erro ao carregar índice FAISS de '{index_path}': {e}")
        raise

def get_or_create_faiss_index_for_specialist(
    specialist_name: str, embeddings: Optional[CohereEmbeddings] = None
) -> FAISS:
    """Obtém ou cria o índice FAISS para um especialista específico."""
    doc_filename = SPECIALIST_DOCUMENTS.get(specialist_name)
    if not doc_filename:
        logger.error(f"Documento não configurado para o especialista: {specialist_name}.")
        raise ValueError(f"Documento não configurado para o especialista: {specialist_name}")

    file_path = os.path.join(DOCS_PATH, doc_filename)
    index_path = SPECIALIST_FAISS_INDEXES[specialist_name]
    
    if embeddings is None:
        embeddings = get_embeddings()

    if os.path.exists(index_path) and os.path.isdir(index_path):
        logger.info(f"Índice FAISS para '{specialist_name}' já existe. Carregando...")
        return load_faiss_index(index_path, embeddings)
    else:
        logger.info(f"Índice FAISS para '{specialist_name}' não encontrado. Criando...")
        documents = load_documents(file_path)
        chunks = split_documents(documents)
        return create_and_save_faiss_index(chunks, index_path, embeddings)

if __name__ == "__main__":
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    logger.info(f"Diretório base para índices FAISS '{FAISS_INDEX_PATH}' verificado/criado.")

    emb = get_embeddings()
    for specialist_name in SPECIALIST_DOCUMENTS.keys():
        try:
            get_or_create_faiss_index_for_specialist(specialist_name, embeddings=emb)
        except Exception as e:
            logger.error(f"Falha crítica ao processar o índice para o especialista '{specialist_name}': {e}")
            
    logger.success("Processamento de índices FAISS para todos os especialistas concluído.")
