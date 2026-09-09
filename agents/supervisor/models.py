"""Definição das estruturas de dados do projeto
Autor: Rodrigo Aguiar
Data: 28/08/2026
"""

from pydantic import BaseModel, Field
from typing import Optional

class AgentSelection(BaseModel):
    """
    Representa a decisão do supervisor sobre qual agente deve processar a query.
    """
    next_agent: str = Field(..., description="O nome do agente especialista selecionado ou 'geral'.")


class TwoAgentSelection(BaseModel):
    """
    Representa a decisão do supervisor sobre até dois especialistas selecionados em ordem de prioridade.
    """
    main_agent: str = Field(..., description="Nome do primeiro agente especialista selecionado ou 'geral'.")
    scnd_agent: Optional[str] = Field(None, description="Nome do segundo agente especialista (peer) ou null/None se não houver.")
