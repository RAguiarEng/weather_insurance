"""core/harness/execution_boundary.py

Camada 3 do Harness: Execution Boundary.

Responsabilidade única deste módulo: decidir se uma rota (especialista)
sugerida pelo Agente Supervisor é válida antes que o grafo LangGraph
efetivamente transicione para ela.

Fonte única de verdade: SPECIALIST_DOCUMENTS (config.py). Este módulo
NÃO lê specialist_summaries.json — esse arquivo é reservado como fonte
de metadados (descrições/resumos) para a Camada 1 (Context Pipeline),
evitando duas fontes de verdade divergentes sobre quais especialistas
realmente existem no sistema.

Este módulo NÃO escolhe a melhor rota (isso é papel do LLM supervisor).
Ele apenas VALIDA e, se necessário, BLOQUEIA uma rota inválida,
substituindo-a por uma rota segura padrão ("geral"), que o grafo já
sabe tratar via route_after_router -> general_handler.

Autor: Rodrigo Aguiar
Data: 01/09/2026
"""

from typing import Optional, Set

from loguru import logger

from config import SPECIALIST_DOCUMENTS

# Rota segura para a qual o sistema cai quando o LLM sugere um destino
# que não existe em SPECIALIST_DOCUMENTS. Não precisa constar no
# conjunto de rotas válidas: route_after_router já trata qualquer valor
# ausente de SPECIALIST_DOCUMENTS como "general_handler".
FALLBACK_ROUTE = "geral"


class RoutingEnforcer:
    """Valida se a rota sugerida pelo supervisor corresponde a um
    especialista real, cadastrado em SPECIALIST_DOCUMENTS.
    """

    def __init__(self, valid_routes: Optional[Set[str]] = None):
        """
        Args:
            valid_routes: conjunto de rotas válidas. Se None, usa as
                chaves de SPECIALIST_DOCUMENTS (comportamento padrão
                em produção). Parâmetro exposto para permitir injeção
                em testes automatizados (Etapa 5) sem depender do
                config.py real.
        """
        self._valid_routes: Set[str] = (
            valid_routes if valid_routes is not None else set(SPECIALIST_DOCUMENTS.keys())
        )
        logger.info(
            "Execution Boundary (RoutingEnforcer) inicializado com {} rotas válidas: {}",
            len(self._valid_routes),
            sorted(self._valid_routes),
        )

    def validate_route(self, suggested_route: Optional[str]) -> str:
        """Verifica se a rota sugerida existe no catálogo de especialistas.

        Args:
            suggested_route: identificador do especialista sugerido
                pelo LLM supervisor (pode vir None, vazio ou inválido).

        Returns:
            A própria rota normalizada, se válida, ou FALLBACK_ROUTE
            caso contrário.
        """
        if not suggested_route:
            logger.warning(
                "Supervisor não retornou rota alguma. Aplicando fallback '{}'.",
                FALLBACK_ROUTE,
            )
            return FALLBACK_ROUTE

        normalized = suggested_route.strip().lower()

        if normalized not in self._valid_routes:
            logger.warning(
                "Rota '{}' sugerida pelo supervisor não existe em SPECIALIST_DOCUMENTS. "
                "Aplicando fallback '{}'.",
                suggested_route,
                FALLBACK_ROUTE,
            )
            return FALLBACK_ROUTE

        return normalized