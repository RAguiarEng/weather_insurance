"""Interface Streamlit - Sistema de Prevenção Climática para Seguradoras
Autor: Rodrigo Aguiar (https://raguiar.eng.br)
Data: 09/09/2026
"""

import streamlit as st
from datetime import datetime
from rag_multiagent import app as langgraph_app

st.set_page_config(
    page_title="Weather Insurance - Prevenção Inteligente",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 20px 25px;
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        border: 1px solid #334155;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h2>🛡️ Weather Insurance AI — Monitoramento Preventivo de Sinistros</h2>
    <p>Detecção antecipada de eventos meteorológicos severos e emissão automatizada de alertas preventivos.</p>
</div>
""", unsafe_allow_html=True)

# Sidebar: Configuração do Segurado e Localização
st.sidebar.header("📍 Parâmetros do Segurado")

MOCK_CLIENTS = {
    "Bruno Corrêa (Porto Alegre - RS)": {
        "client_name": "Bruno Corrêa",
        "city": "Porto Alegre",
        "lat": -30.0330, "lon": -51.2300,
        "policy_type": "Automotivo",
        "policy_id": "AUTO-88219",
        "phone": "+55 51 9888-0011"
    },
    "Jhiovana Ribeiro (Goiânia - GO)": {
        "client_name": "Jhiovana Ribeiro",
        "city": "Goiânia",
        "lat": -16.665136, "lon": -49.286041,
        "policy_type": "Residencial",
        "policy_id": "RES-44312",
        "phone": "+55 61 9999-2233"
    },
    "Luis Pereira (São Paulo - SP)": {
        "client_name": "Luis Pereira",
        "city": "São Paulo",
        "lat": -23.5507, "lon": -46.6334,
        "policy_type": "Residencial",
        "policy_id": "RES-90211",
        "phone": "+55 11 9977-4455"
    },
        "Rodrigo Costa (Belém - PA)": {
        "client_name": "Rodrigo Costa",
        "city": "Belém",
        "lat": -1.45502, "lon": -48.49018,
        "policy_type": "Familiar",
        "policy_id": "FAM-01212",
        "phone": "+55 91 9988-7777"
    },
        "Rodrigo Aguiar (Jaraguá do Sul - SC)": {
        "client_name": "Rodrigo Aguiar",
        "city": "Jaraguá do Sul",
        "lat": -26.485833, "lon": -49.066944,
        "policy_type": "Agro",
        "policy_id": "AGRO-23571",
        "phone": "+55 48 9888-1234"
    },
}

selected_client_key = st.sidebar.selectbox("Selecionar Perfil de Segurado:", list(MOCK_CLIENTS.keys()))
client_data = MOCK_CLIENTS[selected_client_key]

insurance_type = st.sidebar.selectbox("Tipo de Apólice:", ["Automotivo", "Residencial", "Agro", "Familiar"], index=["Automotivo", "Residencial", "Agro", "Familiar"].index(client_data["policy_type"]))

st.sidebar.markdown("---")
st.sidebar.info(f"**Apólice:** {client_data['policy_id']}\n\n**Contato:** {client_data['phone']}")

# Botão de Execução
if st.button("⚡ Executar Monitoramento e Gerar Alerta Preventivo", type="primary"):
    with st.spinner("Coletando dados meteorológicos e processando regras com agentes de IA..."):
        initial_state = {
            "city_name": client_data["city"],
            "coordinates": {"lat": client_data["lat"], "lon": client_data["lon"]},
            "selected_insurance_type": insurance_type,
            "insurance_policy": client_data,
            "messages": []
        }
        
        result = langgraph_app.invoke(initial_state)
        
        weather = result.get("weather_data", {})
        risk = result.get("risk_level", "Baixo")
        event = result.get("weather_event", {})
        
        # Grid 1: Informações Meteorológicas
        st.subheader(f"🌦️ Condições Climáticas em Tempo Real: {client_data['city']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Temperatura", f"{weather.get('temp', 0)} °C")
        c2.metric("Vento", f"{weather.get('wind_speed_kmh', 0)} km/h")
        c3.metric("Chuva (1h)", f"{weather.get('rain_1h_mm', 0)} mm")
        c4.metric("Condição", weather.get("condition_description", "").capitalize())
        
        # Grid 2: Diagnóstico e Regras Acionadas
        st.subheader("⚠️ Diagnóstico de Risco e Regras de Negócio")
        col_risk, col_rules = st.columns([1, 2])
        
        with col_risk:
            st.metric("Nível de Risco Identificado", risk)
            st.write(f"**Eventos:** {', '.join(event.get('events', []))}")
            st.write(f"**Especialista Acionado:** {result.get('selected_specialist')}")
            
        with col_rules:
            st.write("**Regras Acionadas:**")
            for r in result.get("triggered_rules", []):
                st.info(r)
            st.write("**Ações Preventivas Recomendadas:**")
            for a in result.get("preventive_actions", []):
                st.write(f"• {a}")

        # Grid 3: Mensagem Gerada e Simulação de Disparo
        st.subheader("📲 Comunicação Personalizada (Gerada por IA)")
        st.success(result.get("notification_generated"))
        
        st.subheader("📡 Log de Simulação de Disparo")
        st.json(result.get("dispatch_log", {}))
