"""Pipeline LangGraph de Monitoramento Climático e Alerta Preventivo de Seguros
Autor: Rodrigo Aguiar
Data: 09/09/2026
"""

import os
from datetime import datetime
from typing import Dict, Any, List
from loguru import logger
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END

from core.state import AgentState
from core.llm_manager import LLMManager, _call_llm_with_retry
from config import RULES_ENGINE_CONFIG, SPECIALIST_DOCUMENTS
from weather_api_clients.openweather_client import OpenWeatherMapClient
from agents.supervisor.agent import SupervisorAgent
from agents.specialist.base import SpecialistAgent

# Inicialização de Clientes e Agentes
weather_client = OpenWeatherMapClient()
supervisor = SupervisorAgent()
specialist_agents = {name: SpecialistAgent(name=name) for name in SPECIALIST_DOCUMENTS.keys()}

# --- Nó 1: Coleta de Dados Meteorológicos ---
def fetch_weather_node(state: AgentState) -> Dict[str, Any]:
    coords = state.get("coordinates") or {"lat": -23.5505, "lon": -46.6333} # Default: São Paulo
    city = state.get("city_name") or "São Paulo"
    
    logger.info(f"[Nó 1] Coletando clima para {city} ({coords['lat']}, {coords['lon']})...")
    success, raw_current = weather_client.get_current_weather(coords["lat"], coords["lon"])
    _, raw_forecast = weather_client.get_forecast(coords["lat"], coords["lon"])
    
    normalized = weather_client.normalize_current_weather(raw_current) if success else {}
    normalized["city"] = city
    
    return {
        "weather_data": normalized,
        "weather_forecast": raw_forecast or {}
    }

# --- Nó 2: Identificação de Eventos Climáticos de Risco ---
def detect_risks_node(state: AgentState) -> Dict[str, Any]:
    weather = state.get("weather_data", {})
    rain = weather.get("rain_1h_mm", 0.0)
    wind = weather.get("wind_speed_kmh", 0.0)
    condition = weather.get("condition_main", "")
    
    thresholds = RULES_ENGINE_CONFIG
    detected_events = []
    severity = "Baixo"
    
    if rain >= thresholds["rain_mm_threshold"]["severe"] or condition in ["Thunderstorm", "Squall"]:
        detected_events.append("Chuva Torrencial / Tempestade Severa")
        severity = "Crítico"
    elif rain >= thresholds["rain_mm_threshold"]["moderate"]:
        detected_events.append("Chuva Moderada / Risco de Alagamento")
        severity = "Médio" if severity != "Crítico" else severity

    if wind >= thresholds["wind_speed_threshold"]["severe"]:
        detected_events.append("Vendaval / Rajadas Destrutivas")
        severity = "Crítico"
    elif wind >= thresholds["wind_speed_threshold"]["moderate"]:
        detected_events.append("Ventos Fortes")
        severity = "Médio" if severity != "Crítico" else severity

    if condition in thresholds["critical_weather_conditions"]:
        detected_events.append(f"Condição Severa: {condition}")
        severity = "Alto" if severity != "Crítico" else severity

    event_summary = {
        "has_risk": len(detected_events) > 0,
        "events": detected_events if detected_events else ["Condições Estáveis"],
        "severity": severity,
        "timestamp": datetime.now().isoformat()
    }
    
    logger.info(f"[Nó 2] Risco detectado: {event_summary['severity']} - Eventos: {event_summary['events']}")
    return {"weather_event": event_summary, "risk_level": severity}

# --- Nó 3: Aplicação de Regras de Negócio por Apólice ---
def match_policy_rules_node(state: AgentState) -> Dict[str, Any]:
    insurance_type = state.get("selected_insurance_type") or "Residencial"
    risk_level = state.get("risk_level", "Baixo")
    weather = state.get("weather_data", {})
    
    triggered_rules = []
    actions = []
    
    if insurance_type == "Residencial":
        if risk_level in ["Médio", "Alto", "Crítico"]:
            triggered_rules.append("Regra RES-01: Risco de destelhamento, infiltração e queima de aparelhos elétricos.")
            actions.extend([
                "Desconectar aparelhos eletrônicos sensíveis das tomadas.",
                "Verificar fechamento de janelas e desobstrução de calhas.",
                "Evitar permanecer em áreas próximas a árvores de grande porte."
            ])
    elif insurance_type == "Automotivo":
        if risk_level in ["Médio", "Alto", "Crítico"]:
            triggered_rules.append("Regra AUTO-02: Risco de alagamento de vias, queda de galhos e granizo.")
            actions.extend([
                "Estacionar o veículo em local coberto e elevado.",
                "Evitar transitar por vias com histórico de alagamento.",
                "Não tentar atravessar áreas inundadas."
            ])
    elif insurance_type == "Agro":
        if risk_level in ["Médio", "Alto", "Crítico"]:
            triggered_rules.append("Regra AGRO-03: Risco de perda de lavoura por granizo, vendaval ou geada.")
            actions.extend([
                "Proteger maquinários em galpões fechados.",
                "Verificar sistemas de drenagem e contenção de encostas.",
                "Recolher animais para abrigos protegidos."
            ])

    return {
        "triggered_rules": triggered_rules,
        "preventive_actions": actions
    }

# --- Nó 4: Consulta aos Especialistas RAG (FGV / Arruda) ---
def consult_specialist_node(state: AgentState) -> Dict[str, Any]:
    events = state.get("weather_event", {}).get("events", [])
    query = f"Mitigação de perdas e adaptação securitária para: {', '.join(events)}"
    
    routing = supervisor.route(query)
    chosen_specialist = routing.get("selected_agent", "FGV")
    if chosen_specialist not in specialist_agents:
        chosen_specialist = "FGV"
        
    specialist = specialist_agents[chosen_specialist]
    result = specialist.query(query)
    
    return {
        "selected_specialist": chosen_specialist,
        "specialist_response": result["answer"],
        "specialist_confidence": result["confidence"]
    }


# --- Nó 5: Geração de Mensagem Personalizada com LLM ---
def generate_notification_node(state: AgentState) -> Dict[str, Any]:
    llm = LLMManager.get_llm(role="main")
    
    policy = state.get("insurance_policy", {})
    client_name = policy.get("client_name", "Segurado")
    insurance_type = state.get("selected_insurance_type", "Residencial")
    weather = state.get("weather_data", {})
    actions = state.get("preventive_actions", [])
    risk_level = state.get("risk_level", "Baixo")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Você é o Assistente Virtual Proativo de Seguros.
Gere uma mensagem curta, empática e preventiva (estilo WhatsApp / SMS) para o segurado.
Diretrizes:
- Máximo 3 a 4 frases ou tópicos diretos.
- Informe a condição climática prevista na cidade dele.
- Liste 2 a 3 ações preventivas práticas imediatas.
- Conclua com o canal de assistência 24h."""),
        ("human", """Dados:
- Cliente: {client_name}
- Tipo de Seguro: {insurance_type}
- Cidade: {city}
- Condição Climática: {condition} ({temp}°C, Vento: {wind} km/h, Chuva: {rain} mm/h)
- Risco: {risk_level}
- Ações Recomendadas: {actions}""")
    ])
    
    chain = prompt | llm | StrOutputParser()
    message = _call_llm_with_retry(chain, {
        "client_name": client_name,
        "insurance_type": insurance_type,
        "city": weather.get("city", "sua região"),
        "condition": weather.get("condition_description", "Instabilidade"),
        "temp": weather.get("temp", 0),
        "wind": weather.get("wind_speed_kmh", 0),
        "rain": weather.get("rain_1h_mm", 0),
        "risk_level": risk_level,
        "actions": "; ".join(actions)
    })
    
    return {
        "notification_generated": message,
        "notification_channel": "WhatsApp / Push"
    }


# --- Nó 6: Simulação de Disparo ---
def dispatch_simulation_node(state: AgentState) -> Dict[str, Any]:
    policy = state.get("insurance_policy", {})
    log = {
        "status": "DELIVERED_SIMULATED",
        "recipient": policy.get("phone", "+55 11 99999-8888"),
        "channel": state.get("notification_channel", "WhatsApp"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "risk_level": state.get("risk_level", "Baixo")
    }
    return {
        "notification_sent": True,
        "dispatch_log": log
    }

# --- Construção do Grafo ---
workflow = StateGraph(AgentState)
workflow.add_node("fetch_weather", fetch_weather_node)
workflow.add_node("detect_risks", detect_risks_node)
workflow.add_node("match_policy_rules", match_policy_rules_node)
workflow.add_node("consult_specialist", consult_specialist_node)
workflow.add_node("generate_notification", generate_notification_node)
workflow.add_node("dispatch_simulation", dispatch_simulation_node)

workflow.set_entry_point("fetch_weather")
workflow.add_edge("fetch_weather", "detect_risks")
workflow.add_edge("detect_risks", "match_policy_rules")
workflow.add_edge("match_policy_rules", "consult_specialist")
workflow.add_edge("consult_specialist", "generate_notification")
workflow.add_edge("generate_notification", "dispatch_simulation")
workflow.add_edge("dispatch_simulation", END)

app = workflow.compile()
