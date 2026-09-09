"""Gerenciador centralizado de LLMs com fallback em cadeia.

Estratégia de provedores (em ordem de prioridade):
    1. Groq Cloud  — LPU ultrarrápido (~300+ tokens/s), gratuito com limites generosos.
    2. Google Gemini — fallback robusto via langchain-google-genai.
    3. OpenRouter  — fallback final com modelos gratuitos (:free).

Autor: Rodrigo Aguiar.
Data: 31/08/2026
"""

import os
import functools
import requests
import logging
import sys
from typing import List, Any
from loguru import logger

# Importações condicionais: cada uma só é usada se o provedor estiver disponível
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter
from langchain_core.language_models import BaseChatModel # Importar BaseChatModel para type hinting

# Importar tenacity
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import httpx # Para capturar exceções específicas de HTTP

# ---------------------------------------------------------------------------
# Modelos padrão por provedor
# ---------------------------------------------------------------------------

# Groq — prioridade máxima (LPU) - modelos gratuitos confirmados via API
GROQ_MODEL_MAIN: str       = "groq/compound"        # Para agentes especialistas
GROQ_MODEL_SUPERVISOR: str = "groq/compound-mini"   # Para o agente supervisor

# Gemini — fallback intermediário
GEMINI_MODEL_MAIN: str       = "gemini-2.0-flash"         # 131k, ctx, json_mode
GEMINI_MODEL_SUPERVISOR: str = "gemini-2.0-flash"         # mesmo modelo - suporta roteamento json

# OpenRouter — fallback final (modelos gratuitos)
OPENROUTER_MODEL_MAIN: str       = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_MODEL_SUPERVISOR: str = "meta-llama/llama-3.3-70b-instruct:free"


# ---------------------------------------------------------------------------
# Descoberta dinâmica de modelos Groq gratuitos
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _fetch_groq_free_chat_models() -> List[str]:
    """
    Consulta a API Groq e retorna os IDs dos modelos gratuitos aptos para chat/RAG.
    Executa apenas UMA VEZ por sessão — o resultado fica em cache (lru_cache).

    Critérios de seleção (baseados na análise do notebook teste.ipynb):
      - pricing == None  → gratuito (modelos pagos têm pricing definido)
      - 'text' em input_modalities → descarta modelos de áudio (Whisper)
      - context_window >= 8192    → contexto mínimo adequado para RAG
      - 'json_mode' em supported_features → obrigatório para roteamento JSON

    Retorna lista vazia em caso de falha — o modelo padrão hardcoded assume.
    """
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return []

        resp = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5,
        )
        resp.raise_for_status()

        candidates = []
        for m in resp.json().get("data", []):
            pricing    = m.get("pricing")
            input_mod  = m.get("input_modalities", [])
            ctx_window = m.get("context_window", 0)
            features   = m.get("supported_features") or []

            if (
                pricing is None
                and "text" in input_mod
                and ctx_window >= 4000
                and "json_mode" in features
            ):
                candidates.append(m["id"])

        logger.info(f"[LLMManager] Modelos Groq gratuitos disponíveis: {candidates}")
        return candidates

    except Exception as e:
        logger.warning(f"[LLMManager] Falha ao consultar API Groq para descoberta de modelos: {e}")
        return []


# ---------------------------------------------------------------------------
# Decorador de Retry com Exponential Backoff
# ---------------------------------------------------------------------------

# Isso garante que as mensagens de before_sleep_log (que usam o logger padrão)
# sejam formatadas e exibidas pelo loguru.
logger.add(sys.stderr, format="{time} {level} {message}", level="INFO",
           filter="tenacity") # Adiciona um filtro para apenas logs de tenacity
logging.getLogger("tenacity").setLevel(logging.INFO) # Define o nível para o logger de tenacity


# Definir as exceções que devem acionar o retry
# Para Groq, os erros 429 (Rate Limit) e 5xx (Server Error) são os mais comuns.
# LangChain encapsula erros de API em sua própria exceção ou passa a exceção HTTP subjacente.
# httpx.HTTPStatusError é comum para erros de status HTTP.
# Para outros provedores, pode ser necessário adicionar exceções específicas.
RETRY_EXCEPTIONS = (
    httpx.HTTPStatusError, # Erros de status HTTP (4xx, 5xx)
    requests.exceptions.RequestException, # Erros gerais de requisição (conexão, timeout)
    # Adicionar exceções específicas do LangChain ou do provedor se necessário
    # Ex: openai.RateLimitError para OpenAI, google.api_core.exceptions.ResourceExhausted para Gemini
)

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60), # Espera 2^x segundos, min 4s, max 60s
    stop=stop_after_attempt(5), # Tenta no máximo 5 vezes
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    before_sleep=before_sleep_log(logging.getLogger("tenacity"), logging.WARNING),
    reraise=True # Re-lança a exceção se todas as tentativas falharem
)
def _call_llm_with_retry(llm_instance: BaseChatModel, *args: Any, **kwargs: Any) -> Any:
    """
    Função auxiliar para chamar a instância do LLM com lógica de retry.
    """
    return llm_instance.invoke(*args, **kwargs)


class LLMManager:
    """
    Fábrica de LLMs com fallback automático entre provedores.

    Uso:
        llm = LLMManager.get_llm(role="main")
        llm = LLMManager.get_llm(role="supervisor")

    Parâmetros:
        role (str): "main" para agentes especialistas, "supervisor" para o agente supervisor.

    Retorna:
        Uma instância de LLM compatível com LangChain (ChatGroq, ChatGoogleGenerativeAI
        ou ChatOpenRouter), na primeira opção disponível.

    Lança:
        RuntimeError: Se nenhum provedor estiver configurado corretamente.
    """

    @staticmethod
    def get_llm(role: str = "main"):
        """
        Tenta instanciar o LLM na ordem: Groq → Gemini → OpenRouter.
        
        Args:
            role: "main" (especialistas) ou "supervisor".
        
        Returns:
            Instância de BaseChatModel pronta para uso.
        """
        # Seleciona os modelos corretos para cada papel
        if role == "supervisor":
            groq_model       = GROQ_MODEL_SUPERVISOR
            gemini_model     = GEMINI_MODEL_SUPERVISOR
            openrouter_model = OPENROUTER_MODEL_SUPERVISOR
        else:
            groq_model       = GROQ_MODEL_MAIN
            gemini_model     = GEMINI_MODEL_MAIN
            openrouter_model = OPENROUTER_MODEL_MAIN

        # ------------------------------------------------------------------
        # Tentativa 1: Groq Cloud
        # ------------------------------------------------------------------
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key:
            try:
                # Descobre dinamicamente o melhor modelo gratuito disponível;
                # usa o modelo padrão hardcoded se a descoberta falhar ou retornar lista vazia.
                free_models = _fetch_groq_free_chat_models()
                selected_groq_model = free_models[0] if free_models else groq_model

                # Instancia o LLM
                llm = ChatGroq(
                    model=selected_groq_model,
                    api_key=groq_api_key,
                    temperature=0,
                )
                logger.info(
                    f"[LLMManager] Provedor ativo: Groq | "
                    f"Modelo: {selected_groq_model} | Role: {role}"
                )
                return llm
            except Exception as e:
                logger.warning(f"[LLMManager] Groq indisponível: {e}. Tentando Gemini...")
        else:
            logger.warning("[LLMManager] GROQ_API_KEY não encontrada. Pulando Groq.")

        # ------------------------------------------------------------------
        # Tentativa 2: Google Gemini
        # ------------------------------------------------------------------
        gemini_api_key = os.getenv("GOOGLE_API_KEY")
        if gemini_api_key:
            try:
                llm = ChatGoogleGenerativeAI(
                    model=gemini_model,
                    google_api_key=gemini_api_key,
                    temperature=0,
                )
                logger.info(f"[LLMManager] Provedor ativo: Gemini | Modelo: {gemini_model} | Role: {role}")
                return llm
            except Exception as e:
                logger.warning(f"[LLMManager] Gemini indisponível: {e}. Tentando OpenRouter...")
        else:
            logger.warning("[LLMManager] GOOGLE_API_KEY / GEMINI_API_KEY não encontradas. Pulando Gemini.")

        # ------------------------------------------------------------------
        # Tentativa 3: OpenRouter (fallback final)
        # ------------------------------------------------------------------
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_api_key:
            try:
                llm = ChatOpenRouter(
                    model=openrouter_model,
                    api_key=openrouter_api_key,
                )
                logger.info(f"[LLMManager] Provedor ativo: OpenRouter | Modelo: {openrouter_model} | Role: {role}")
                return llm
            except Exception as e:
                logger.error(f"[LLMManager] OpenRouter também falhou: {e}.")

        # ------------------------------------------------------------------
        # Nenhum provedor disponível
        # ------------------------------------------------------------------
        raise RuntimeError(
            "[LLMManager] Nenhum provedor LLM está disponível. "
            "Verifique as variáveis GROQ_API_KEY, GOOGLE_API_KEY e OPENROUTER_API_KEY no arquivo .env."
        )
