"""Interface Streamlit - Sistema de Prevenção Climática para Seguradoras (Layout em Abas)
Autor: Rodrigo Aguiar (https://raguiar.eng.br)
Data: 09/09/2026
"""

import json
from datetime import datetime
import streamlit as st
from rag_multiagent import app as langgraph_app
from config import RULES_ENGINE_CONFIG, OPENWEATHERMAP_API_KEY
import folium
from streamlit_folium import st_folium

# --- Configurações da Página ---
st.set_page_config(
    page_title="Weather Insurance AI — Central de Prevenção",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Estilos CSS Customizados (UI/UX Profissional & Alto Contraste) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 20px 26px;
        color: #f8fafc;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.25);
    }
    
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.95rem;
        font-weight: 700;
        margin: 0 0 4px 0;
        color: #ffffff;
    }
    
    /* Card de Boas-vindas / Placeholder */
    .placeholder-card {
        background: #1e293b;
        border: 1px dashed #475569;
        border-radius: 14px;
        padding: 35px 25px;
        text-align: center;
        color: #94a3b8;
        margin-top: 20px;
    }
    
    /* Cards das Etapas Sequenciais */
    .stage-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
    }
    
    .stage-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 14px;
        border-bottom: 1px solid #334155;
        padding-bottom: 10px;
    }
    
    .stage-title {
        font-family: 'Outfit', sans-serif !important;
        color: #f8fafc !important;
        font-size: 1.18rem !important;
        font-weight: 600 !important;
        margin: 0 !important;
    }
    
    .stage-badge {
        background: #0284c7;
        color: #ffffff;
        font-size: 0.75rem;
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 9999px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Smartphone Mockup */
    .phone-container {
        background: #090d16;
        border: 2px solid #334155;
        border-radius: 24px;
        padding: 18px;
        box-shadow: 0 20px 35px -10px rgba(0, 0, 0, 0.4);
        max-width: 480px;
        margin: 0 auto;
    }
    
    .phone-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 8px;
        margin-bottom: 12px;
        font-size: 0.78rem;
        color: #94a3b8;
    }
    
    .whatsapp-bubble {
        background: #064e3b;
        color: #f0fdf4;
        border-radius: 14px 14px 2px 14px;
        padding: 15px 18px;
        font-size: 0.93rem;
        line-height: 1.55;
        border: 1px solid #059669;
    }
    
    .bubble-footer {
        display: flex;
        justify-content: flex-end;
        align-items: center;
        gap: 4px;
        font-size: 0.72rem;
        color: #6ee7b7;
        margin-top: 6px;
    }
    
    /* Badges de Severidade */
    .risk-badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 14px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
    }
    .risk-critico { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .risk-alto { background: rgba(249, 115, 22, 0.2); color: #fb923c; border: 1px solid #f97316; }
    .risk-medio { background: rgba(234, 179, 8, 0.2); color: #facc15; border: 1px solid #eab308; }
    .risk-baixo { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid #22c55e; }
</style>
""", unsafe_allow_html=True)

# --- Banner Superior ---
st.markdown("""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1 class="hero-title">🛡️ Weather Insurance AI</h1>
            <p style="color: #94a3b8; margin: 0; font-size: 1.0rem;">
                Monitoramento climático preventivo de sinistros com IA Generativa e Agentes Inteligentes.
            </p>
        </div>
        <div>
            <span style="background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.3); padding: 6px 14px; border-radius: 9999px; font-size: 0.8rem; font-weight: 600;">
                ● LangGraph StateGraph Ativo
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# --- Base de Segurados Cadastrados ---
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
        "lat": -16.6651, "lon": -49.2860,
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
        "lat": -1.4550, "lon": -48.4901,
        "policy_type": "Familiar",
        "policy_id": "FAM-01212",
        "phone": "+55 91 9988-7777"
    },
    "Rodrigo Aguiar (Jaraguá do Sul - SC)": {
        "client_name": "Rodrigo Aguiar",
        "city": "Jaraguá do Sul",
        "lat": -26.4858, "lon": -49.0669,
        "policy_type": "Agro",
        "policy_id": "AGRO-23571",
        "phone": "+55 48 9888-1234",
        "email": "raguiar.eng@gmail.com"
    },
}

# --- Sidebar: Seleção do Segurado ---
st.sidebar.header("👤 1. Seleção do Segurado")

selected_client_key = st.sidebar.selectbox("Segurado Cadastrado:", list(MOCK_CLIENTS.keys()))
client_data = MOCK_CLIENTS[selected_client_key]

insurance_type = st.sidebar.selectbox(
    "Tipo de Apólice:",
    ["Automotivo", "Residencial", "Agro", "Familiar"],
    index=["Automotivo", "Residencial", "Agro", "Familiar"].index(client_data["policy_type"])
)

st.sidebar.markdown("---")
st.sidebar.header("🌦️ 2. Modo de Coleta Meteorológica")
weather_mode = st.sidebar.radio(
    "Fonte de Dados:",
    [
        "API Real (OpenWeatherMap)",
        "Simular: Tempestade Severa com Granizo",
        "Simular: Vendaval e Chuva Torrencial",
        "Simular: Condição Estável (Sem Risco)"
    ]
)

# Atualização visual imediata dos dados cadastrais do cliente
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Dados Cadastrais")
st.sidebar.info(
    f"• **Apólice:** `{client_data['policy_id']}`\n\n"
    f"• **Titular:** {client_data['client_name']}\n\n"
    f"• **Cobertura Selecionada:** `{insurance_type}`"
)

# --- Gerenciamento de Estado para Limpar a Tela ao Mudar o Segurado ---
current_selection_key = f"{selected_client_key}_{insurance_type}_{weather_mode}"

if "last_selection_key" not in st.session_state:
    st.session_state.last_selection_key = current_selection_key
    st.session_state.current_result = None

# Se o usuário trocou o segurado ou qualquer opção, limpa os resultados da tela
if st.session_state.last_selection_key != current_selection_key:
    st.session_state.current_result = None
    st.session_state.last_selection_key = current_selection_key

# --- Função de Execução do Workflow ---
def execute_workflow():
    initial_state = {
        "city_name": client_data["city"],
        "coordinates": {"lat": client_data["lat"], "lon": client_data["lon"]},
        "selected_insurance_type": insurance_type,
        "insurance_policy": client_data,
        "messages": []
    }
    
    result = langgraph_app.invoke(initial_state)
    
    if weather_mode == "Simular: Tempestade Severa com Granizo":
        result["weather_data"] = {
            "city": client_data["city"],
            "temp": 18.5,
            "feels_like": 17.0,
            "humidity": 92,
            "wind_speed_kmh": 68.4,
            "rain_1h_mm": 42.0,
            "condition_main": "Thunderstorm",
            "condition_description": "Tempestade com granizo e chuva torrencial"
        }
        result["risk_level"] = "Crítico"
        result["weather_event"] = {
            "has_risk": True,
            "events": ["Chuva Torrencial / Tempestade Severa", "Vendaval / Rajadas Destrutivas", "Condição Severa: Granizo"],
            "severity": "Crítico",
            "timestamp": datetime.now().isoformat()
        }
    elif weather_mode == "Simular: Vendaval e Chuva Torrencial":
        result["weather_data"] = {
            "city": client_data["city"],
            "temp": 21.0,
            "feels_like": 20.0,
            "humidity": 88,
            "wind_speed_kmh": 72.0,
            "rain_1h_mm": 36.5,
            "condition_main": "Squall",
            "condition_description": "Rajadas de vento intensas e chuva forte"
        }
        result["risk_level"] = "Crítico"
        result["weather_event"] = {
            "has_risk": True,
            "events": ["Chuva Torrencial / Tempestade Severa", "Vendaval / Rajadas Destrutivas"],
            "severity": "Crítico",
            "timestamp": datetime.now().isoformat()
        }
    elif weather_mode == "Simular: Condição Estável (Sem Risco)":
        result["weather_data"] = {
            "city": client_data["city"],
            "temp": 24.0,
            "feels_like": 24.0,
            "humidity": 55,
            "wind_speed_kmh": 12.0,
            "rain_1h_mm": 0.0,
            "condition_main": "Clear",
            "condition_description": "Céu limpo e tempo estável"
        }
        result["risk_level"] = "Baixo"
        result["weather_event"] = {
            "has_risk": False,
            "events": ["Condições Estáveis"],
            "severity": "Baixo",
            "timestamp": datetime.now().isoformat()
        }
        result["triggered_rules"] = ["Nenhuma regra emergencial ativada. Monitoramento de rotina ativo."]
        result["preventive_actions"] = ["Manter acompanhamento periódico."]
        
    return result

# Botão na barra lateral
if st.sidebar.button("🔄 Atualizar e Executar Pipeline", type="primary", use_container_width=True):
    with st.spinner("Consultando dados meteorológicos e acionando agentes de IA..."):
        st.session_state.current_result = execute_workflow()

# --- Renderização do Conteúdo Principal ---
if st.session_state.get("current_result") is None:
    st.markdown(f"""
    <div class="placeholder-card">
        <h3 style="color: #f8fafc; font-family: 'Outfit', sans-serif; margin-bottom: 8px;">
            📍 Segurado Selecionado: {client_data['client_name']} ({client_data['city']})
        </h3>
        <p style="font-size: 0.95rem; margin-bottom: 20px;">
            Apólice: <strong>{client_data['policy_id']}</strong> | Cobertura: <strong>{insurance_type}</strong> | Modo: <strong>{weather_mode}</strong>
        </p>
        <p style="color: #38bdf8; font-weight: 600; font-size: 1.05rem;">
            👈 Clique no botão "Atualizar e Executar Pipeline" no menu lateral para iniciar o processamento.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    result = st.session_state.current_result
    weather = result.get("weather_data", {})
    risk = result.get("risk_level", "Baixo")
    event = result.get("weather_event", {})
    specialist_name = result.get("selected_specialist", "FGV")
    notification_text = result.get("notification_generated", "")
    dispatch_log = result.get("dispatch_log", {})

    risk_css_map = {
        "Crítico": "risk-critico",
        "Alto": "risk-alto",
        "Médio": "risk-medio",
        "Baixo": "risk-baixo"
    }
    risk_class = risk_css_map.get(risk, "risk-baixo")

    # --- Criação das Abas ---
    tab_principal, tab_etapas = st.tabs([
        "📲 Central de Comunicação & Telemetria",
        "🔄 Fluxo Detalhado das 5 Etapas"
    ])

    # =========================================================================
    # ABA 1: CENTRAL DE COMUNICAÇÃO & TELEMETRIA (PRINCIPAL / DEFAULT)
    # =========================================================================
    with tab_principal:
        # --- Bloco de Aviso de Risco (Nova Implementação) ---
        # A mensagem será exibida sempre, com conteúdo condicional.
        if event.get("has_risk", False) and risk in ["Crítico", "Alto", "Médio"]:
            eventos_detectados = event.get("events", ["condições climáticas adversas"])
            # Formata a lista de eventos para uma string legível
            evento_relevante_str = ", ".join(eventos_detectados)

            st.warning(
                f"**{client_data['client_name']}**, atenção! Em sua região está prevista **{evento_relevante_str}**."
            )
        else:
            # Mensagem padrão quando não há risco relevante para o segurado selecionado
            st.info(
                f"Monitoramento contínuo: As condições climáticas para o segurado **{client_data['client_name']}** "
                f"({client_data['city']}) estão estáveis e sem riscos relevantes no momento. "
                f"Nenhuma ação preventiva é necessária."
            )
        st.markdown("<br>", unsafe_allow_html=True) # Adiciona um pequeno espaçamento após o aviso

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Segurado", client_data["client_name"], f"Apólice: {insurance_type}")
        m2.metric("Localidade", client_data["city"], f"{weather.get('temp', 0):.1f} °C · {weather.get('condition_description', '')}")
        m3.metric("Severidade Climática", risk, f"Vento: {weather.get('wind_speed_kmh', 0):.1f} km/h")
        m4.metric("Canal de Envio", result.get("notification_channel", "WhatsApp"), "Status: Entregue")

        st.markdown("<br>", unsafe_allow_html=True)

        # Colunas para o mockup do celular e o mapa
        col_mockup, col_map = st.columns([1.1, 1], gap="large") # Ajuste os pesos conforme desejar

        with col_mockup:
            st.subheader("📱 Simulação Visual no Dispositivo do Segurado")
            st.caption("Notificação preventiva enviada antes da ocorrência do sinistro.")

            st.markdown(f"""
            <div class="phone-container">
                <div class="phone-header">
                    <span>💬 <strong>Seguradora Clima 24h</strong></span>
                    <span>{datetime.now().strftime('%H:%M')}</span>
                </div>
                <div class="whatsapp-bubble">
                    {notification_text.replace(chr(10), '<br>')}
                </div>
                <div class="bubble-footer">
                    <span>✓✓ Entregue via WhatsApp / Push</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_map:
            st.subheader("🗺️ Localização do Risco")
            st.caption("Visualização geográfica da apólice do segurado e camadas climáticas.")

            # Cria o mapa Leaflet centrado no cliente
            mapa = folium.Map(location=[client_data["lat"], client_data["lon"]], zoom_start=12)

            # Adiciona o marcador do cliente
            folium.Marker(
                [client_data["lat"], client_data["lon"]], 
                popup=client_data["client_name"],
                tooltip=f"{client_data['client_name']} - {insurance_type}"
            ).add_to(mapa)

            # Adiciona a camada base do OpenStreetMap (para ter um fundo)
            folium.TileLayer(
                tiles="OpenStreetMap",
                name="Mapa Base (OpenStreetMap)",
                control=True
            ).add_to(mapa)

            # Adiciona as camadas visuais da OpenWeatherMap
            # Certifique-se de que OPENWEATHERMAP_API_KEY está importado de config.py
            from config import OPENWEATHERMAP_API_KEY # Adicione esta linha no topo do app.py se ainda não o fez
            if OPENWEATHERMAP_API_KEY:
                # Camada de Precipitação
                folium.TileLayer(
                    tiles=f"https://tile.openweathermap.org/map/precipitation_new/{{z}}/{{x}}/{{y}}.png?appid={OPENWEATHERMAP_API_KEY}",
                    attr="Map data &copy; OpenWeatherMap",
                    name="Radar de Precipitação",
                    overlay=True,
                    control=True,
                    opacity=0.6
                ).add_to(mapa)

                # Camada de Nuvens
                folium.TileLayer(
                    tiles=f"https://tile.openweathermap.org/map/clouds_new/{{z}}/{{x}}/{{y}}.png?appid={OPENWEATHERMAP_API_KEY}",
                    attr="Map data &copy; OpenWeatherMap",
                    name="Cobertura de Nuvens",
                    overlay=True,
                    control=True,
                    opacity=0.5
                ).add_to(mapa)

                # Camada de Temperatura
                folium.TileLayer(
                    tiles=f"https://tile.openweathermap.org/map/temp_new/{{z}}/{{x}}/{{y}}.png?appid={OPENWEATHERMAP_API_KEY}",
                    attr="Map data &copy; OpenWeatherMap",
                    name="Temperatura",
                    overlay=True,
                    control=True,
                    opacity=0.7
                ).add_to(mapa)

                # Camada de Vento
                folium.TileLayer(
                    tiles=f"https://tile.openweathermap.org/map/wind_new/{{z}}/{{x}}/{{y}}.png?appid={OPENWEATHERMAP_API_KEY}",
                    attr="Map data &copy; OpenWeatherMap",
                    name="Velocidade do Vento",
                    overlay=True,
                    control=True,
                    opacity=0.6
                ).add_to(mapa)

                # Adiciona o controle de camadas no canto do mapa
                folium.LayerControl().add_to(mapa)

            # Renderiza o mapa Leaflet dentro do Streamlit
            st_folium(mapa, width="100%", height=400, key=f"map_{client_data['policy_id']}")

        # A coluna de telemetria agora fica em uma nova linha, abaixo das colunas anteriores
        st.markdown("<br>", unsafe_allow_html=True) # Adiciona um espaço
        col_telemetry = st.container() # Cria um container para a telemetria na próxima linha

        with col_telemetry:
            st.subheader("📡 Telemetria e Log de Disparo Simulado")
            st.caption("Registro estruturado de auditoria e confirmação de entrega multicanal.")

            st.json({
                "event_id": f"EVT-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "delivery_status": "DELIVERED_SIMULATED",
                "recipient_name": client_data["client_name"],
                "recipient_phone": client_data["phone"],
                "city": client_data["city"],
                "policy_id": client_data["policy_id"],
                "policy_type": insurance_type,
                "risk_severity": risk,
                "events_detected": event.get("events", []),
                "specialist_consulted": specialist_name,
                "channel": result.get("notification_channel", "WhatsApp / Push"),
                "dispatch_timestamp": dispatch_log.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "actions_suggested_count": len(result.get("preventive_actions", []))
            })
            
    # =========================================================================
    # ABA 2: FLUXO DETALHADO DAS ETAPAS SEQUENCIAIS
    # =========================================================================
    with tab_etapas:
        st.caption("Detalhamento arquitetural da execução de ponta a ponta pelo grafo LangGraph.")
        
        # ETAPA 1
        st.markdown(f"""
        <div class="stage-card">
            <div class="stage-header">
                <span class="stage-badge">Etapa 1</span>
                <h3 class="stage-title">📡 Coleta de Dados Meteorológicos (OpenWeatherMap API)</h3>
            </div>
        """, unsafe_allow_html=True)
        
        e1_c1, e1_c2, e1_c3, e1_c4 = st.columns(4)
        e1_c1.metric("Temperatura", f"{weather.get('temp', 0):.1f} °C", f"Sensação: {weather.get('feels_like', weather.get('temp', 0)):.1f} °C")
        e1_c2.metric("Vento / Rajadas", f"{weather.get('wind_speed_kmh', 0):.1f} km/h")
        e1_c3.metric("Volume de Chuva (1h)", f"{weather.get('rain_1h_mm', 0):.1f} mm")
        e1_c4.metric("Condição Geral", str(weather.get("condition_description", "")).capitalize())
        
        with st.expander("🔍 Inspecionar Payload Normalizado da API"):
            st.json(weather)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # ETAPA 2
        st.markdown(f"""
        <div class="stage-card">
            <div class="stage-header">
                <span class="stage-badge">Etapa 2</span>
                <h3 class="stage-title">⚠️ Identificação de Eventos Climáticos Relevantes</h3>
        </div>
        """, unsafe_allow_html=True)
        
        e2_c1, e2_c2 = st.columns([1, 2])
        with e2_c1:
            st.markdown("**Nível de Risco Classificado:**")
            st.markdown(f"<span class='risk-badge {risk_class}'>● {risk}</span>", unsafe_allow_html=True)
            st.markdown("<br>**Eventos Detectados:**", unsafe_allow_html=True)
            for ev in event.get("events", ["Condições Estáveis"]):
                st.write(f"• **{ev}**")
                
        with e2_c2:
            st.markdown("**Limiares Paramétricos de Referência (`RULES_ENGINE_CONFIG`):**")
            st.write(f"• **Chuva:** Moderada > {RULES_ENGINE_CONFIG['rain_mm_threshold']['moderate']} mm/h | Severa > {RULES_ENGINE_CONFIG['rain_mm_threshold']['severe']} mm/h")
            st.write(f"• **Vento:** Forte > {RULES_ENGINE_CONFIG['wind_speed_threshold']['moderate']} km/h | Vendaval > {RULES_ENGINE_CONFIG['wind_speed_threshold']['severe']} km/h")
            st.write(f"• **Condições Críticas Monitoradas:** {', '.join(RULES_ENGINE_CONFIG['critical_weather_conditions'])}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # ETAPA 3
        st.markdown(f"""
        <div class="stage-card">
            <div class="stage-header">
                <span class="stage-badge">Etapa 3</span>
                <h3 class="stage-title">📋 Aplicação de Regras de Negócio e Apólices</h3>
            </div>
        """, unsafe_allow_html=True)
        
        e3_c1, e3_c2 = st.columns([1, 2])
        with e3_c1:
            st.write(f"**Apólice:** `{client_data['policy_id']}`")
            st.write(f"**Modalidade:** `{insurance_type}`")
            st.write(f"**Município:** {client_data['city']}")
        with e3_c2:
            st.write("**Regras Securitárias Ativadas:**")
            for rule in result.get("triggered_rules", []):
                st.info(rule)
            st.write("**Recomendações Preventivas Aplicáveis:**")
            for act in result.get("preventive_actions", []):
                st.write(f"✅ {act}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        # ETAPA 4
        st.markdown(f"""
        <div class="stage-card">
            <div class="stage-header">
                <span class="stage-badge">Etapa 4</span>
                <h3 class="stage-title">📚 Consulta Especializada RAG (Literatura de Desastres e Seguros)</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.write(f"**Agente Especialista Roteado:** `{specialist_name}` *(Base Documental Indexada via FAISS e Cohere Embeddings)*")
        with st.expander(f"📖 Parecer do Especialista ({specialist_name})", expanded=True):
            st.write(result.get("specialist_response", "Diretrizes técnicas gerais de proteção patrimonial aplicadas."))
        st.markdown("</div>", unsafe_allow_html=True)
        
        # ETAPA 5
        st.markdown(f"""
        <div class="stage-card">
            <div class="stage-header">
                <span class="stage-badge">Etapa 5</span>
                <h3 class="stage-title">💬 Geração da Comunicação Personalizada (IA Generativa)</h3>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("**Mensagem Sintetizada pelo LLM:**")
        st.success(notification_text)
        st.markdown("</div>", unsafe_allow_html=True)
