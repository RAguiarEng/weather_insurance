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
        system_message = f"""Você é o especialista técnico em regulação e impacto de riscos climáticos em seguros: {self.name}.
Utilize o contexto documental recuperado abaixo para embasar sua resposta e formular recomendações técnicas de prevenção e adaptação a desastres climáticos.

Contexto Recuperado:
{retrieved_context}

Diretrizes:
1. Responda com clareza, objetividade e rigor técnico.
2. Foque em medidas de mitigação de danos, adaptação e princípios securitários.
3. Se o contexto não contiver informações suficientes, mencione claramente as boas práticas gerais de proteção patrimonial."""

        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", "{query}")
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
