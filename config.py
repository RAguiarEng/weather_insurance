"""Configurações centrais do projeto
Autor: Rodrigo Aguiar
Data: 09/09/2026
"""

# Sistema
import os
from dotenv import load_dotenv

load_dotenv()

# --- Configurações de LangSmith ---
LANGSMITH_TRACING: str | None = os.getenv("LANGSMITH_TRACING")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT")

# --- Configurações de API --- 
OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
COHERE_API_KEY: str | None = os.getenv("COHERE_API_KEY")
OPENWEATHERMAP_API_KEY: str | None = os.getenv("OPENWEATHERMAP_API_KEY")

# --- Configurações da OpenWeatherMap ---
OPENWEATHERMAP_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
OPENWEATHERMAP_FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
OPENWEATHERMAP_DEFAULT_LANG: str = "pt_br"
OPENWEATHERMAP_DEFAULT_UNITS: str = "metric"
OPENWEATHERMAP_MAX_REQUESTS_PER_HOUR: int = 40
OPENWEATHERMAP_TIMEOUT_SECONDS: int = 30
OPENWEATHERMAP_MAX_RETRIES: int = 3

# --- Configurações de Modelos ---
# Modelo de embeddings para vetorização de documentos e queries
EMBEDDING_MODEL: str = 'embed-multilingual-v3.0'                         # via COHERE

# Modelos LLM - gerenciados pelo core/llm_manager.py
# A seleção automática segue a ordem: Groq -> Gemini -> OpenRouter
# Estas constantes são mantidas apenas como referência/documentação:
LLM_MAIN: str = 'groq/compound'                                # via Groq (padrão)
LLM_SUPERVISOR: str = 'groq/compound-min'                      # via Groq (padrão)

# Modelo de LLM leve para reescrita de query
LLM_REWRITE: str = 'gemma3:1b'                                           # modelo local

# --- Configurações de RAG ---
# Tamanho dos pedaços de texto ao dividir os documentos
CHUNK_SIZE: int = 500
# Tamanho da sobreposição entre os pedaços para manter contexto
CHUNK_OVERLAP: int = 100
# Qde de pedaços a serem recuperados por busca
TOP_K_RETRIEVAL: int = 3

# --- Caminhos de Diretórios ---
# Caminho base para os documentos PDF
DOCS_PATH: str = "docs/weather"
# Caminho base para salvar os índices FAISS
FAISS_INDEX_PATH: str = "faiss_index"

# --- Configurações dos Agentes Especialistas --- 
# Mapeia o nome descritivo de cada especialista ao nome do arquivo PDF que ele gerencia.
SPECIALIST_DOCUMENTS: dict = {
    "FGV": "FGV_Seguro_Mudancas_Climaticas.pdf",
    "Arruda": "ARRUDA_Seguro_Catastrofes_Climaticas.pdf"
}

# Gera automaticamente os caminhos completos para os diretórios dos índices FAISS de cada especialista.
SPECIALIST_FAISS_INDEXES: dict = {
    name: os.path.join(FAISS_INDEX_PATH, f"{name}_faiss_index")
    for name in SPECIALIST_DOCUMENTS.keys()
}

CLIMATE_SPECIALISTS: list = list(SPECIALIST_DOCUMENTS.keys())
# --- Regras e Limiares do Motor de Risco Meteorológico ---
RULES_ENGINE_CONFIG: dict = {
    "rain_mm_threshold": {
        "moderate": 15.0,   # mm/h
        "severe": 35.0      # mm/h
    },
    "wind_speed_threshold": {
        "moderate": 40.0,   # km/h
        "severe": 65.0      # km/h
    },
    "temp_threshold": {
        "heatwave": 38.0,   # °C
        "frost": 3.0        # °C (risco de geada para agro)
    },
    "critical_weather_conditions": [
        "Thunderstorm", "Squall", "Tornado", "Hail"
    ]
}
