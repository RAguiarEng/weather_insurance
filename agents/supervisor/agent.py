"""Estrutura do Agente Supervisor com LangGraph para Seguros e Clima
Autor: Rodrigo Aguiar
Data: 09/09/2026
"""

import os
import json
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from loguru import logger

from config import SPECIALIST_DOCUMENTS
from core.llm_manager import LLMManager, _call_llm_with_retry

class SupervisorAgent:
    """Supervisor que analisa eventos climáticos, direciona para os especialistas FGV/Arruda
    e consolida orientações técnicas e de mitigação de risco."""
    def __init__(self):
        self.llm = LLMManager.get_llm(role='supervisor')

        summaries_path = os.path.join(os.path.dirname(__file__), "specialist_summaries.json")
        try:
            with open(summaries_path, 'r', encoding='utf-8') as f:
                self.specialist_summaries_data = json.load(f)
            self.specialist_summaries_map = {item['agent']: item for item in self.specialist_summaries_data}
            logger.info("Resumos dos especialistas de seguros carregados com sucesso.")
        except FileNotFoundError:
            logger.error(f"Arquivo '{summaries_path}' não encontrado.")
            self.specialist_summaries_data = []
            self.specialist_summaries_map = {}

        self.specialist_names = list(self.specialist_summaries_map.keys())
        self.full_context_str = self._format_all_summaries()

        # Cadeia de Roteamento
        self.router_prompt = self._build_router_prompt()
        self.router_chain = self.router_prompt | self.llm | JsonOutputParser()

        # Cadeia de Avaliação Geral
        self.general_eval_prompt = self._build_general_eval_prompt()
        self.general_eval_chain = self.general_eval_prompt | self.llm | StrOutputParser()

        logger.success("Agente Supervisor de Seguros Clima inicializado com sucesso.")

    def _format_all_summaries(self) -> str:
        parts = []
        for item in self.specialist_summaries_data:
            parts.append(f"### Especialista: '{item.get('agent')}'")
            parts.append(f"- Descrição: {item.get('description')}")
            if item.get("trigger_examples"):
                parts.append(f"- Temas de Acionamento: {', '.join(item['trigger_examples'])}")
        return "\n".join(parts)


    def _build_router_prompt(self) -> ChatPromptTemplate:
        system_template = """Você é o Classificador de Roteamento de Seguros e Riscos Climáticos.
Sua única tarefa é escolher o especialista mais adequado ('FGV' ou 'Arruda') com base no evento informado.

Especialistas:
- FGV: Foco em gestão de riscos climáticos, regulação e adaptação preventiva.
- Arruda: Foco em catástrofes, sinistros severos e cobertura jurídica de seguros.

Retorne EXCLUSIVAMENTE um objeto JSON válido (sem textos extras ou markdown antes/depois):
{{
  "selected_agent": "FGV ou Arruda",
  "confidence": 0.9,
  "reasoning": "Justificativa em 1 frase curta."
}}"""
        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "Evento / Consulta: {query}")
        ])


    def _build_general_eval_prompt(self) -> ChatPromptTemplate:
        system_template = """Você é o Supervisor de Inteligência de Sinistros e Clima.
Seu papel é sintetizar os dados de risco meteorológico e as orientações dos especialistas para formular uma diretriz preventiva de mitigação de danos para a seguradora."""
        return ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", "{query}")
        ])

    def route(self, query: str) -> Dict[str, Any]:
        """Executa a decisão de roteamento."""
        try:
            response = _call_llm_with_retry(self.router_chain, {"query": query})
            return response
        except Exception as e:
            logger.error(f"Erro no roteamento do Supervisor: {e}")
            return {"selected_agent": self.specialist_names[0] if self.specialist_names else "FGV", "confidence": 0.5, "reasoning": "Fallback de erro"}
