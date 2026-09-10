"""Lógica base dos Agentes Especialistas de Seguros Clima
Autor: Rodrigo Aguiar
Data: 09/09/2026
"""

import os
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from core.llm_manager import LLMManager, _call_llm_with_retry
from indexing import get_embeddings, get_or_create_faiss_index_for_specialist
from config import TOP_K_RETRIEVAL

class SpecialistAgent:
    """Agente especialista em literatura técnica de seguros e catástrofes climáticas (FGV / Arruda)."""
    def __init__(self, name: str):
        self.name = name
        logger.info(f"Inicializando Agente Especialista '{self.name}'...")
        self.embeddings = get_embeddings()
        self.vector_store = get_or_create_faiss_index_for_specialist(name, self.embeddings)
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K_RETRIEVAL})
        self.llm = LLMManager.get_llm(role="main")
        logger.success(f"Agente Especialista '{self.name}' pronto.")

    def _build_prompt(self, retrieved_context: str) -> ChatPromptTemplate:
        system_message = f"""Você é o consultor técnico de seguros '{self.name}'.
Seu papel é extrair do contexto documental apenas diretrizes preventivas práticas e imediatas para proteção patrimonial e mitigação de perdas do segurado.

Contexto Documental:
{retrieved_context}

Regras Obrigatórias de Resposta:
1. Seja extremamente CONCISO e DIRETO ao ponto (máximo 3 a 4 tópicos curtos).
2. Não faça introduções longas, históricos ou teses teóricas.
3. Foque apenas em ações práticas aplicáveis à segurança do bem segurado."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "Evento Climático / Risco: {query}")
        ])


    def query(self, query_text: str) -> Dict[str, Any]:
        """Recupera documentos via RAG e gera a análise do especialista."""
        try:
            docs = self.retriever.invoke(query_text)
            context_text = "\n\n".join([doc.page_content for doc in docs])
            prompt = self._build_prompt(context_text)
            chain = prompt | self.llm | StrOutputParser()
            answer = _call_llm_with_retry(chain, {"query": query_text})
            
            return {
                "answer": answer,
                "sources": [doc.metadata.get("source", self.name) for doc in docs],
                "confidence": 0.85 if docs else 0.4
            }
        except Exception as e:
            logger.error(f"Erro na execução do especialista '{self.name}': {e}")
            return {"answer": f"Não foi possível consultar a base do especialista {self.name}.", "sources": [], "confidence": 0.0}
