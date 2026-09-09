"""core/harness/context_pipeline.py

Camada 2 do Harness: Context Pipeline - Token Budgeter.

Responsabilidade única deste módulo: truncar o histórico de mensagens
(AgentState["messages"]) para caber em um orçamento de tokens configurável,
antes de ser injetado em qualquer prompt. Isso garante fluidez conversacional
(o agente "lembra" das últimas trocas) sem permitir crescimento não
controlado de custo por requisição à medida que a conversa avança.

Usa trim_messages (langchain_core.messages), com um token_counter simples
baseado em heurística de caracteres (~4 caracteres por token), consistente
com o fallback já usado em extract_token_usage (rag_multiagent.py). Essa
escolha evita depender de um tokenizer específico de provedor — Groq,
Gemini e OpenRouter não expõem uma forma unificada e gratuita de contagem
exata de tokens — e evita o custo de uma chamada de API só para medir
tamanho de contexto.

Autor: Rodrigo Aguiar
Data: 01/09/2026
"""

from typing import List

from langchain_core.messages import BaseMessage, trim_messages
from loguru import logger

# Orçamento padrão de tokens para o histórico de conversa injetado no
# prompt. Deliberadamente conservador: o objetivo é preservar contexto
# recente o suficiente para fluidez conversacional, sem permitir que o
# histórico cresça de forma proporcional (e cara) ao tamanho da conversa.
# Ajustável por instância, caso um agente específico precise de um
# orçamento diferente no futuro.
DEFAULT_HISTORY_TOKEN_BUDGET = 1000


def _approximate_token_counter(messages: List[BaseMessage]) -> int:
    """
    Heurística simples de contagem de tokens (~4 caracteres por token),
    usada apenas para decidir o corte do histórico. Não substitui a
    contagem real de tokens reportada pelo provedor (usage_metadata),
    que já é tratada separadamente em extract_token_usage.
    """
    total_chars = sum(len(m.content or "") for m in messages)
    return max(1, total_chars // 4)


class TokenBudgeter:
    """
    Trunca o histórico de mensagens para caber em um orçamento de tokens,
    preservando sempre as trocas mais recentes (estratégia "last").
    """

    def __init__(self, max_tokens: int = DEFAULT_HISTORY_TOKEN_BUDGET):
        self.max_tokens = max_tokens

    def trim(self, messages: List[BaseMessage]) -> List[BaseMessage]:
        """
        Retorna uma versão truncada de `messages`, respeitando o
        orçamento de tokens configurado.

        Args:
            messages: histórico completo de mensagens da conversa.

        Returns:
            Lista truncada, mantendo as mensagens mais recentes.
        """
        if not messages:
            return messages

        original_count = len(messages)
        trimmed = trim_messages(
            messages,
            max_tokens=self.max_tokens,
            token_counter=_approximate_token_counter,
            strategy="last",
            include_system=True,
            allow_partial=False,
        )

        if len(trimmed) < original_count:
            logger.info(
                "TokenBudgeter: histórico truncado de {} para {} mensagens "
                "(orçamento: ~{} tokens).",
                original_count,
                len(trimmed),
                self.max_tokens,
            )

        return trimmed