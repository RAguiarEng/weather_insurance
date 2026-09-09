"""Definição do AgentState para o LangGraph
Autor: Rodrigo Aguiar
Data: 01/09/2026
"""

from typing import List, TypedDict, Annotated, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """
    Representa o estado global da conversa no LangGraph.
     Reflete a memória hierárquica e o novo fluxo híbrido de comunicação A2A.
    """
    user_query: Optional[str]                               # A pergunta original do usuário, opcional para monitoramento do tempo. 
    current_agent: Optional[str]                            # O agente que está processando a requisição no momento.
    messages: Annotated[List[BaseMessage], add_messages]    # Histórico de mensagens.
    agent_responses: List[Dict[str, Any]]                   # Respostas parciais ou finais dos agentes especialistas.
    next_agent: Optional[str]                               # Nome do primeiro especialista selecionado pelo supervisor.
    selected_specialist: Optional[str]                      # Nome do primeiro especialista.
    second_specialist: Optional[str]                        # Nome do segundo especialista (peer).
    specialist_confidence: Optional[float]                  # Confiança do especialista retornado.
    final_answer: Optional[str]                             # A resposta final consolidada pelo supervisor.
    specialist_response: Optional[str]                      # Resposta do especialista (se houver).
    error_message: Optional[str]                            # Mensagens de erro ou falha.
    context_id: Optional[str]                               # ID da sessão para rastreamento.

    # Métricas de cada etapa
    supervisor_tokens: Optional[int]
    supervisor_cost: Optional[float]
    supervisor_latency: Optional[float]

    specialist_tokens: Optional[int]
    specialist_cost: Optional[float]
    specialist_latency: Optional[float]

    # Métricas totais
    total_tokens: Optional[int]
    total_cost: Optional[float]
    total_latency: Optional[float]

    # Observabilidade e rastreamento
    trace_id: Optional[str]
    start_time: Optional[float]
    end_time: Optional[float]
    recursion_limit_counter: Optional[int]

    # Campos para o fluxo "Supervisor-First"
    supervisor_attempted: Optional[bool]    # True se o supervisor tentou responder via specialist_summaries.json
    supervisor_confident: Optional[bool]    # True se o supervisor obteve resposta confiável pelo JSON

    # Campos para colaboração peer-to-peer (A2A)
    peer_exchange_log: Optional[List[Dict[str, Any]]]   # Log das mensagens A2A trocadas
    peer_turn_count: Optional[Dict[str, int]]           # Ex: {"agenteA-agenteB": 2} (máximo 2 por par)
    peer_final_response: Optional[str]                  # Resposta consolidada após troca entre especialistas

    # Sinais do Harness (Camadas 3 e 5). Registro mínimo de auditoria:
    # indica se a Camada 5 (Verification Loop) revogou uma confiança
    # falsamente positiva, ou se a Camada 3 (Execution Boundary) precisou
    # corrigir uma rota inválida sugerida pelo LLM. A consolidação formal
    # desses sinais em um trilha de auditoria completa é escopo da Etapa 4
    # (Memory/State + Observabilidade).
    verification_override: Optional[bool]                   # True se StrictConfidenceVerifier revogou a confiança do supervisor
    routing_override: Optional[bool]                        # True se RoutingEnforcer corrigiu a rota sugerida pelo supervisor
    specialist_confidence_override: Optional[bool]          # True se StrictConfidenceVerifier corrigiu a confiança numérica de um especialista/peer
    groundedness_override: Optional[bool]                   # True se StrictConfidenceVerifier detectou dado de contato fabricado (e-mail/telefone/CNPJ) na resposta do supervisor
 
    # Camada 2 do Harness (Context Pipeline): histórico de conversa já
    # truncado por TokenBudgeter, calculado uma vez por invocação do grafo
    # (nó prepare_context) e injetado nos prompts que precisam de contexto
    # conversacional. O campo messages permanece como o registro completo
    # e não-truncado da conversa (reducer add_messages).
    budgeted_history: Optional[List[BaseMessage]]

    # --- Dados de Monitoramento e Seguros (MVP de Clima) ---
    city_name: Optional[str]                                # Nome da cidade monitorada
    coordinates: Optional[Dict[str, float]]                 # {'lat': float, 'lon': float}
    weather_data: Optional[Dict[str, Any]]                  # Dados brutos normalizados da API
    weather_forecast: Optional[Dict[str, Any]]              # Dados de previsão futura
    weather_event: Optional[Dict[str, Any]]                 # Evento crítico detectado (tipo, severidade, métricas)
    insurance_policy: Optional[Dict[str, Any]]              # Dados da apólice do segurado
    selected_insurance_type: Optional[str]                  # 'Residencial', 'Automotivo', 'Agro'
    risk_level: Optional[str]                               # 'Baixo', 'Médio', 'Alto', 'Crítico'
    triggered_rules: Optional[List[str]]                    # Regras de negócio acionadas
    preventive_actions: Optional[List[str]]                 # Recomendações técnicas de prevenção
    notification_generated: Optional[str]                   # Mensagem personalizada gerada via LLM
    notification_channel: Optional[str]                     # 'WhatsApp', 'SMS', 'E-mail', 'Push'
    notification_sent: Optional[bool]                       # Flag de simulação de envio
    dispatch_log: Optional[Dict[str, Any]]                  # Metadados e timestamp do disparo simulado
