"""Gerenciador centralizado de LLMs com fallback em tempo de execução (Runtime Fallback).

Estratégia de provedores com alternância automática em tempo de requisição:
    1. Groq Cloud   — Modelos de código aberto com inferência ultra-rápida (gpt-oss-120b / gpt-oss-20b).
    2. Google Gemini — Fallback de estabilidade alta (gemini-3.6-flash).
    3. OpenRouter   — Fallback final de contingência (:free).

Autor: Rodrigo Aguiar.
Data de Revisão: 10/09/2026
"""

import os
import functools
import requests
from typing import List, Any
from loguru import logger

# LangChain Chat Models
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable

# ---------------------------------------------------------------------------
# Modelos padrão por provedor (Validados e Ativos)
# ---------------------------------------------------------------------------

# Groq (Modelos open-source rápidos e gratuitos no LPU)
GROQ_MODEL_MAIN: str = "openai/gpt-oss-120b"       # Especialistas e Notificação
GROQ_MODEL_SUPERVISOR: str = "openai/gpt-oss-20b"   # Supervisor / Roteamento JSON

# Google Gemini (Versão atual da API)
GEMINI_MODEL_MAIN: str = "gemini-3.6-flash"
GEMINI_MODEL_SUPERVISOR: str = "gemini-3.6-flash"

# OpenRouter (Modelos gratuitos)
OPENROUTER_MODEL_MAIN: str = "meta-llama/llama-3.3-70b-instruct:free"
OPENROUTER_MODEL_SUPERVISOR: str = "meta-llama/llama-3.3-70b-instruct:free"


# ---------------------------------------------------------------------------
# Descoberta dinâmica de modelos Groq
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _fetch_groq_free_chat_models() -> List[str]:
    """Consulta a API Groq e retorna os IDs dos modelos de chat disponíveis."""
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
            model_id = m.get("id", "")
            # Exclui modelos de áudio e proteção de prompt
            if (
                "whisper" not in model_id.lower()
                and "prompt-guard" not in model_id.lower()
                and "orpheus" not in model_id.lower()
            ):
                candidates.append(model_id)

        # Ordena colocando os modelos gpt-oss e qwen no topo
        candidates.sort(key=lambda x: "120b" in x or "20b" in x or "qwen" in x, reverse=True)

        logger.info(f"[LLMManager] Modelos Groq selecionados do catálogo: {candidates}")
        return candidates

    except Exception as e:
        logger.warning(f"[LLMManager] Falha ao consultar catálogo da Groq: {e}")
        return []


# ---------------------------------------------------------------------------
# Função de Execução Resiliente
# ---------------------------------------------------------------------------

def _call_llm_with_retry(runnable: Runnable, input_data: Any, **kwargs: Any) -> Any:
    """Executa a invocação de um Runnable com fallback automático."""
    try:
        return runnable.invoke(input_data, **kwargs)
    except Exception as e:
        logger.error(f"[LLMManager] Todos os provedores (incluindo fallbacks) falharam: {e}")
        raise e


# ---------------------------------------------------------------------------
# Gerenciador de LLMs com Fallback em Runtime
# ---------------------------------------------------------------------------

class LLMManager:
    """Fábrica de LLMs com cadeia de fallback ativa em tempo de execução."""

    @staticmethod
    def _create_groq_llm(role: str) -> BaseChatModel | None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            return None
        try:
            free_models = _fetch_groq_free_chat_models()
            default_model = GROQ_MODEL_SUPERVISOR if role == "supervisor" else GROQ_MODEL_MAIN
            selected_model = default_model if default_model in free_models else (free_models[0] if free_models else default_model)

            return ChatGroq(
                model=selected_model,
                api_key=groq_api_key,
                temperature=0,
                max_retries=1,
            )
        except Exception as e:
            logger.warning(f"[LLMManager] Não foi possível preparar o cliente Groq: {e}")
            return None

    @staticmethod
    def _create_gemini_llm(role: str) -> BaseChatModel | None:
        gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            return None
        try:
            model_name = GEMINI_MODEL_SUPERVISOR if role == "supervisor" else GEMINI_MODEL_MAIN
            return ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=gemini_api_key,
                temperature=0,
                max_retries=1,
            )
        except Exception as e:
            logger.warning(f"[LLMManager] Não foi possível preparar o cliente Gemini: {e}")
            return None

    @staticmethod
    def _create_openrouter_llm(role: str) -> BaseChatModel | None:
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        if not openrouter_api_key:
            return None
        try:
            model_name = OPENROUTER_MODEL_SUPERVISOR if role == "supervisor" else OPENROUTER_MODEL_MAIN
            return ChatOpenRouter(
                model=model_name,
                api_key=openrouter_api_key,
            )
        except Exception as e:
            logger.warning(f"[LLMManager] Não foi possível preparar o cliente OpenRouter: {e}")
            return None

    @classmethod
    def get_llm(cls, role: str = "main") -> Runnable:
        """Monta uma cadeia de fallback em tempo de execução (Groq -> Gemini -> OpenRouter)."""
        available_llms: List[BaseChatModel] = []

        # 1. Groq
        groq_llm = cls._create_groq_llm(role)
        if groq_llm:
            available_llms.append(groq_llm)

        # 2. Gemini
        gemini_llm = cls._create_gemini_llm(role)
        if gemini_llm:
            available_llms.append(gemini_llm)

        # 3. OpenRouter
        openrouter_llm = cls._create_openrouter_llm(role)
        if openrouter_llm:
            available_llms.append(openrouter_llm)

        if not available_llms:
            raise RuntimeError(
                "[LLMManager] Nenhum provedor LLM configurado. "
                "Defina GROQ_API_KEY, GOOGLE_API_KEY ou OPENROUTER_API_KEY no arquivo .env."
            )

        primary_llm = available_llms[0]
        fallbacks = available_llms[1:]

        if fallbacks:
            logger.info(
                f"[LLMManager] LLM montado para role '{role}' com "
                f"Provedor Principal: {primary_llm.__class__.__name__} e {len(fallbacks)} Fallback(s) em runtime."
            )
            return primary_llm.with_fallbacks(fallbacks=fallbacks)

        logger.info(f"[LLMManager] LLM montado para role '{role}' com provedor único: {primary_llm.__class__.__name__}")
        return primary_llm
