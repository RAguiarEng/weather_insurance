"""Definição dos protocolos de comunicação ACPMessage e A2AMessage
Autor: Rodrigo Aguiar
Data: 09/09/2026
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Protocol, Tuple

class ACPMessage(BaseModel):
    """
    Protocolo de Comunicação entre Agentes (ACP - Agent Communication Protocol).
    Define a estrutura padronizada para mensagens trocadas entre agentes e o supervisor.
    """
    sender: str = Field(..., description="Identificador do agente remetente.")
    main_receiver: str = Field(..., description="Identificador do principal agente destinatário (ex: 'supervisor' ou nome do especialista).")
    scnd_receiver: Optional[str] = Field(None, description="Segundo agente especialista prioritário selecionado pelo supervisor.")
    receiver: Optional[str] = Field(None, description="Alias mantido para compatibilidade retroativa.")
    intent: str = Field(..., description="A intenção da mensagem (ex: 'query_retrieval', 'final_answer', 'clarification').")

    # Dicionário flexível com os dados da mensagem
    payload: Dict[str, Any] = Field(default_factory=dict, description="Conteúdo da mensagem, varia conforme a intenção.")

    context_id: Optional[str] = Field("session_default", description="ID da sessão ou contexto da conversa para rastreamento.")
    timestamp: str = Field(..., description="Timestamp da mensagem no formato ISO 8601.")

    confidence: Optional[float] = Field(None, description="Nível de confiança da informação no payload (0.0 a 1.0).")
    sources: Optional[List[str]] = Field(None, description="Lista de fontes ou documentos utilizados.")

    def to_dict(self) -> Dict[str, Any]:
        """Converte a mensagem ACP para um dicionário."""
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ACPMessage":
        """Cria uma instância de ACPMessage a partir de um dicionário."""
        return cls(**data)

    def __str__(self):
        return f"ACPMessage(sender={self.sender}, main_receiver={self.main_receiver}, scnd_receiver={self.scnd_receiver}, intent={self.intent}, context_id={self.context_id})"


class A2AMessage(BaseModel):
    """
    Protocolo de Comunicação Agente-a-Agente (A2A - Agent-to-Agent Protocol).
    Define a estrutura de mensagens trocadas diretamente entre agentes especialistas,
    sem passar pela interação com o usuário.

    Regras de uso:
    - Um mesmo par de agentes pode trocar mensagens no máximo 2 vezes.
    - A última mensagem da troca deve ser encaminhada ao supervisor (via ACPMessage).
    - O supervisor é o único agente autorizado a interagir com o usuário.
    """
    sender: str = Field(..., description="Identificador do agente especialista remetente.")
    main_receiver: str = Field(..., description="Identificador do agente especialista destinatário principal.")
    scnd_receiver: Optional[str] = Field(None, description="Segundo agente parceiro ou supervisor.")
    receiver: Optional[str] = Field(None, description="Alias mantido para compatibilidade retroativa.")
    intent: str = Field(..., description="Intenção da mensagem entre pares: 'peer_query' (A→B) ou 'peer_response' (B→A).")
    
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Conteúdo da mensagem: pergunta original, resposta parcial, documentos recuperados."
    )
    context_id: Optional[str] = Field("session_default", description="ID da sessão para rastreamento.")
    timestamp: str = Field(..., description="Timestamp ISO 8601.")
    turn: int = Field(..., description="Número do turno desta troca entre o par (1 = A→B, 2 = B→A). Máximo: 2.")
    confidence: Optional[float] = Field(None, description="Confiança da resposta (0.0 a 1.0).")
    sources: Optional[List[str]] = Field(None, description="Fontes utilizadas.")

    def is_final_turn(self) -> bool:
        """Retorna True se este é o último turno permitido entre o par."""
        return self.turn >= 2

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        return cls(**data)

    def __str__(self):
        return (
            f"A2AMessage(sender={self.sender}, main_receiver={self.main_receiver}, "
            f"intent={self.intent}, turn={self.turn}/2, context_id={self.context_id})"
        )


class AgentResponse(BaseModel):
    """
    Modelo para padronizar a resposta de um agente, incluindo o payload
    (resposta principal) e um nível de confiança.
    """
    payload: Dict[str, Any]
    confidence: float = 0.0 # Nível de confiança da resposta (0.0 a 1.0)
    # Adicione outros campos se necessário, como tokens_used, latency, etc.


class WeatherApiClient(Protocol):
    """Protocolo estrito para clientes de APIs meteorológicas externas."""
    def get_current_weather(
        self, lat: float, lon: float, extra_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Obtém dados meteorológicos atuais."""
        ...
    
    def get_forecast(
        self, lat: float, lon: float, days: int = 5, extra_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Obtém previsão do tempo para os próximos dias/períodos."""