import json
from datetime import datetime

import streamlit as st
import folium
from streamlit_folium import st_folium
import httpx
from weather_api_clients.openweather_client import OpenWeatherMapClient

from config import RULES_ENGINE_CONFIG, OPENWEATHERMAP_API_KEY
from rag_multiagent import app as langgraph_app

owm_client = OpenWeatherMapClient()
weather_client_app = OpenWeatherMapClient()

# --- Funções Auxiliares para Simulação de Risco (Determinística) ---
# Estas funções simulam a lógica de detecção de risco do pipeline,
# mas de forma leve para a carga inicial do Streamlit.

def _get_risk_score(risk_level: str) -> int:
    """Atribui um score numérico a cada nível de risco."""
    if risk_level == "Crítico":
        return 4
    elif risk_level == "Alto":
        return 3
    elif risk_level == "Médio":
        return 2
    elif risk_level == "Baixo":
        return 1
    return 0

def _detect_simulated_risk(weather_data: dict, weather_client_instance: OpenWeatherMapClient) -> dict:
    """Simula a detecção de risco com base em dados meteorológicos."""
    risk_level = "Baixo"
    events = ["Condições Estáveis"]
    has_risk = False

    # Lógica simplificada baseada em RULES_ENGINE_CONFIG
    rain_mm = weather_data.get("rain_1h_mm", 0)
    wind_speed = weather_data.get("wind_speed_kmh", 0)
    temp = weather_data.get("temp", 0)
    condition_main = weather_data.get("condition_main", "")

    if rain_mm >= RULES_ENGINE_CONFIG["rain_mm_threshold"]["severe"]:
        risk_level = "Crítico"
        events.append("Chuva Torrencial / Tempestade Severa")
        has_risk = True
    elif rain_mm >= RULES_ENGINE_CONFIG["rain_mm_threshold"]["moderate"]:
        if _get_risk_score(risk_level) < _get_risk_score("Médio"):
            risk_level = "Médio"
        events.append("Chuva Forte")
        has_risk = True

    if wind_speed >= RULES_ENGINE_CONFIG["wind_speed_threshold"]["severe"]:
        if _get_risk_score(risk_level) < _get_risk_score("Crítico"):
            risk_level = "Crítico"
        events.append("Vendaval / Rajadas Destrutivas")
        has_risk = True
    elif wind_speed >= RULES_ENGINE_CONFIG["wind_speed_threshold"]["moderate"]:
        if _get_risk_score(risk_level) < _get_risk_score("Alto"):
            risk_level = "Alto"
        events.append("Vento Forte")
        has_risk = True

    if temp >= RULES_ENGINE_CONFIG["temp_threshold"]["heatwave"]:
        if _get_risk_score(risk_level) < _get_risk_score("Alto"):
            risk_level = "Alto"
        events.append("Onda de Calor Extrema")
        has_risk = True
    elif temp <= RULES_ENGINE_CONFIG["temp_threshold"]["frost"]:
        if _get_risk_score(risk_level) < _get_risk_score("Alto"):
            risk_level = "Alto"
        events.append("Risco de Geada")
        has_risk = True

    if condition_main in RULES_ENGINE_CONFIG["critical_weather_conditions"]:
        if _get_risk_score(risk_level) < _get_risk_score("Crítico"):
            risk_level = "Crítico"
        events.append(f"Condição Severa: {condition_main}")
        has_risk = True

    # Remove eventos duplicados e "Condições Estáveis" se houver risco real
    if has_risk:
        events = list(set(e for e in events if e != "Condições Estáveis"))
    else:
        events = ["Condições Estáveis"]

    return {
        "has_risk": has_risk,
        "events": events,
        "severity": risk_level,
        "timestamp": datetime.now().isoformat()
    }

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
        "phone": "+55 48 9888-1234"
    },
    "Carlos Daniel (Palma - ESP)": {
        "client_name": "Carlos Daniel",
        "city": "Palma",
        "lat": 39.571314, "lon": 2.651651,
        "policy_type": "Residencial",
        "policy_id": "RES-78990",
        "phone": "+34 971 8888-7777"
    }
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
    st.session_state.initial_risk_assessment = None # Novo estado para avaliação inicial

# Se o usuário trocou o segurado ou qualquer opção, limpa os resultados da tela
if st.session_state.last_selection_key != current_selection_key:
    st.session_state.current_result = None
    st.session_state.initial_risk_assessment = None # Limpa também a avaliação inicial
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

    # --- Lógica para INJETAR dados simulados no initial_state ---
    if weather_mode == "Simular: Tempestade Severa com Granizo":
        initial_state["weather_data"] = {
            "city": client_data["city"],
            "temp": 18.5,
            "feels_like": 17.0,
            "humidity": 92,
            "wind_speed_kmh": 68.4,
            "rain_1h_mm": 42.0,
            "condition_main": "Thunderstorm",
            "condition_description": "Tempestade com granizo e chuva torrencial"
        }
        initial_state["risk_level"] = "Crítico"
        initial_state["weather_event"] = {
            "has_risk": True,
            "events": ["Chuva Torrencial / Tempestade Severa", "Vendaval / Rajadas Destrutivas", "Condição Severa: Granizo"],
            "severity": "Crítico",
            "timestamp": datetime.now().isoformat()
        }
        # Podemos pré-definir algumas regras e ações para simulação, se quisermos
        initial_state["triggered_rules"] = ["Regra AUTO-02: Risco de alagamento de vias, queda de galhos e granizo."]
        initial_state["preventive_actions"] = ["Estacionar o veículo em local coberto e elevado.", "Evitar transitar por vias com histórico de alagamento."]

    elif weather_mode == "Simular: Vendaval e Chuva Torrencial":
        initial_state["weather_data"] = {
            "city": client_data["city"],
            "temp": 21.0,
            "feels_like": 20.0,
            "humidity": 88,
            "wind_speed_kmh": 72.0,
            "rain_1h_mm": 36.5,
            "condition_main": "Squall",
            "condition_description": "Rajadas de vento intensas e chuva forte"
        }
        initial_state["risk_level"] = "Crítico"
        initial_state["weather_event"] = {
            "has_risk": True,
            "events": ["Chuva Torrencial / Tempestade Severa", "Vendaval / Rajadas Destrutivas"],
            "severity": "Crítico",
            "timestamp": datetime.now().isoformat()
        }
        initial_state["triggered_rules"] = ["Regra RES-01: Risco de destelhamento, infiltração e queima de aparelhos elétricos."]
        initial_state["preventive_actions"] = ["Desconectar aparelhos eletrônicos sensíveis das tomadas.", "Verificar fechamento de janelas e desobstrução de calhas."]

    elif weather_mode == "Simular: Condição Estável (Sem Risco)":
        initial_state["weather_data"] = {
            "city": client_data["city"],
            "temp": 24.0,
            "feels_like": 24.0,
            "humidity": 55,
            "wind_speed_kmh": 12.0,
            "rain_1h_mm": 0.0,
            "condition_main": "Clear",
            "condition_description": "Céu limpo e tempo estável"
        }
        initial_state["risk_level"] = "Baixo"
        initial_state["weather_event"] = {
            "has_risk": False,
            "events": ["Condições Estáveis"],
            "severity": "Baixo",
            "timestamp": datetime.now().isoformat()
        }
        initial_state["triggered_rules"] = ["Nenhuma regra emergencial ativada. Monitoramento de rotina ativo."]
        initial_state["preventive_actions"] = ["Manter acompanhamento periódico."]

    result = langgraph_app.invoke(initial_state)

    return result

# --- Lógica de Avaliação de Risco Inicial (ao carregar a página) ---
if st.session_state.initial_risk_assessment is None:
    highest_risk_client = None
    highest_risk_score = -1
    highest_risk_event = None
    highest_risk_weather = None


    for client_key, client_info in MOCK_CLIENTS.items():
        try:
            # Simula a coleta de dados meteorológicos para cada cliente
            success, raw_current_weather = weather_client_app.get_current_weather( # Use weather_client_app
                lat=client_info["lat"],
                lon=client_info["lon"]
            )

            # Normaliza os dados brutos, assim como é feito no rag_multiagent.py
            # E adiciona a cidade para consistência
            normalized_weather_data = weather_client_app.normalize_current_weather(raw_current_weather) if success else {} # Use weather_client_app
            normalized_weather_data["city"] = client_info["city"] # Adiciona a cidade para _detect_simulated_risk

            # Simula a detecção de risco usando os dados normalizados
            simulated_event = _detect_simulated_risk(normalized_weather_data, weather_client_app)

            simulated_risk_level = simulated_event["severity"]

            current_score = _get_risk_score(simulated_risk_level)

            if current_score > highest_risk_score:
                highest_risk_score = current_score
                highest_risk_client = client_info
                highest_risk_event = simulated_event
                highest_risk_weather = normalized_weather_data
        except Exception as e:
            # Em caso de erro na API (ex: chave inválida), ignora o cliente para a avaliação inicial
            # e loga o erro, mas não impede o carregamento da página.
            print(f"Erro ao buscar clima para {client_info['client_name']}: {e}")
            continue

    st.session_state.initial_risk_assessment = {
        "client": highest_risk_client,
        "event": highest_risk_event,
        "weather": highest_risk_weather,
        "risk_score": highest_risk_score
    }

# Botão na barra lateral
if st.sidebar.button("🔄 Atualizar e Executar Pipeline", type="primary", use_container_width=True):
    with st.spinner("Consultando dados meteorológicos e acionando agentes de IA..."):
        st.session_state.current_result = execute_workflow()

# --- Renderização do Conteúdo Principal ---
# Bloco de aviso inicial para a seguradora (sempre visível ao carregar a página)
if st.session_state.initial_risk_assessment:
    initial_client = st.session_state.initial_risk_assessment["client"]
    initial_event = st.session_state.initial_risk_assessment["event"]
    initial_risk_score = st.session_state.initial_risk_assessment["risk_score"]

    if initial_risk_score > _get_risk_score("Baixo"): # Se houver algum risco relevante
        eventos_detectados = initial_event.get("events", ["condições climáticas adversas"])
        evento_relevante_str = ", ".join(eventos_detectados)
        st.error( # Usando st.error para maior destaque no aviso inicial
            f"**ALERTA DE PRIORIDADE:** O segurado **{initial_client['client_name']}** "
            f"({initial_client['city']}) apresenta a condição mais crítica no momento: "
            f"**{evento_relevante_str}**."
        )
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.success( # Usando st.success para mensagem de tranquilidade inicial
            "**STATUS GERAL:** Todos os segurados estão sob monitoramento e, no momento, "
            "não há riscos climáticos relevantes detectados para nenhum deles."
        )
        st.markdown("<br>", unsafe_allow_html=True)


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
        # --- Bloco de Aviso de Risco (Após execução do pipeline para o cliente selecionado) ---
        if event.get("has_risk", False) and risk in ["Crítico", "Alto", "Médio"]:
            eventos_detectados = event.get("events", ["condições climáticas adversas"])
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