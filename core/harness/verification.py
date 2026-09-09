"""core/harness/verification.py

Camada 5 do Harness: Verification Loop.

Responsabilidade deste módulo: revalidar sinais de confiança declarados
pelos agentes (Supervisor e Especialistas) antes que sejam usados para
decidir se uma resposta é definitiva ou se o fluxo deve continuar
(roteamento, colaboração A2A), e detectar fabricação grosseira de dados
de contato que não existem na fonte permitida.

Pontos de origem de bug já identificados e corrigidos por este módulo:

1. SupervisorAgent.try_answer_from_summaries() decide confiança de forma
   puramente sintática (content.endswith("[CONFIANTE]")), permitindo que
   uma resposta que na verdade admite incerteza seja tratada como definitiva.
   -> corrigido por verify().

2. SpecialistAgent.invoke() decide confiança de forma binária, checando
   apenas três frases fixas ("não sei", "não consta", "não foi encontrada").
   -> corrigido por verify_specialist_confidence().

3. SupervisorAgent.try_answer_from_summaries() usa apenas short_context_str
   (agent + description) como fonte, mas o LLM pode alucinar dados de
   contato específicos (e-mail, telefone, CNPJ) que não existem nessa
   fonte, mesmo quando a resposta não contém nenhum padrão de incerteza
   textual (é uma alucinação "confiante", não uma admissão de dúvida).
   -> tratado por verify_groundedness(). Esta é uma checagem determinística
   e propositalmente limitada (dados de contato); avaliação de fidelidade
   semântica mais ampla (ex.: listas de documentos plausíveis mas
   inventadas) é escopo da suíte de regressão automatizada (Etapa 5),
   usando a métrica de faithfulness da biblioteca ragas (já presente em
   requirements.txt), não regex.

Autor: Rodrigo Aguiar
"""

import re
from typing import Any, Dict, List, Tuple

from loguru import logger

# Padrões (case-insensitive) que indicam que a resposta, mesmo marcada
# como confiante (textual ou numericamente) pelo agente, está na verdade
# admitindo incerteza, insuficiência de informação, ou delegando a decisão
# a um especialista/humano. Lista deliberadamente extensível: novos padrões
# observados em produção (Etapa 5: suíte de regressão) devem ser
# adicionados aqui. Esta é a fonte única de verdade para detecção de
# incerteza textual, usada tanto para o Supervisor quanto para os
# Especialistas.
#
# NOTA TÉCNICA: evitar escrever colchetes escapados (|$$ $$|) como texto
# literal em regex nesta lista. Editores/renderizadores de Markdown ou
# LaTeX podem interpretar essa sequência como delimitador de bloco de
# fórmula e corrompê-la ao copiar/colar. A tag literal "[INSUFICIENTE]"
# é tratada separadamente abaixo, via re.escape() sobre uma string
# comum, sem barras invertidas visíveis no código-fonte.
UNCERTAINTY_PATTERNS: List[str] = [
    r"consultar(?:emos| a| o)? (?:o |a )?especialista",
    r"n[ãa]o (?:tenho|temos) certeza",
    r"n[ãa]o (?:foi|é) poss[íi]vel (?:confirmar|encontrar)",
    r"n[ãa]o encontrei (?:essa |esta )?informa[çc][ãa]o",
    r"n[ãa]o h[áa] informa[çc][ãa]o espec[íi]fica",
    r"recomendo (?:verificar|consultar)",
    r"seria necess[áa]rio (?:verificar|consultar)",
    r"n[ãa]o (?:consta|est[áa]) nos (?:manuais|documentos)",
    r"n[ãa]o (?:sei|consta|foi encontrada)",
]

# Tag literal tratada isoladamente, sem escrever colchetes escapados
# diretamente na lista acima.
_INSUFFICIENT_TAG = "[INSUFICIENTE]"

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in UNCERTAINTY_PATTERNS]
_COMPILED_PATTERNS.append(re.compile(re.escape(_INSUFFICIENT_TAG), re.IGNORECASE))

# Confiança atribuída quando um padrão de incerteza é detectado no texto
# de um especialista, mas a heurística numérica original (base.py) havia
# retornado um valor alto. Escolhido abaixo do limiar de 0.7 usado em
# maybe_a2a_exchange (rag_multiagent.py), para garantir que a colaboração
# A2A seja corretamente acionada nesses casos.
DEFAULT_LOW_CONFIDENCE = 0.3

# Padrões de dados de contato/identificação que NUNCA deveriam aparecer
# em uma resposta gerada a partir de short_context_str (apenas
# agent + description), já que essa fonte não contém esse tipo de dado.
# Presença desses padrões na resposta, sem correspondência literal na
# fonte fornecida, é tratada como fabricação (groundedness violation).
# NOTA TÉCNICA: o padrão de telefone evita escrever parênteses escapados
# (|$ $|) como texto literal no código-fonte, pelo mesmo motivo já
# documentado para _INSUFFICIENT_TAG acima — editores/renderizadores
# podem corromper essa sequência ao copiar/colar. Os parênteses opcionais
# do DDD são construídos via re.escape() sobre caracteres comuns.
_OPEN_PAREN = re.escape("(")
_CLOSE_PAREN = re.escape(")")
_PHONE_PATTERN = rf"{_OPEN_PAREN}?\d{{2}}{_CLOSE_PAREN}?\s?\d{{4,5}}-?\d{{4}}"

CONTACT_INFO_PATTERNS: Dict[str, "re.Pattern"] = {
    "email": re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
    "telefone": re.compile(_PHONE_PATTERN),
    "cnpj": re.compile(r"\d{2}\.?\d{3}\.?\d{3}/\d{4}-?\d{2}"),
}


class StrictConfidenceVerifier:
    """
    Revalida semanticamente sinais de confiança (textuais ou numéricos)
    declarados pelos agentes, e detecta fabricação grosseira de dados de
    contato que não existem na fonte permitida, corrigindo falsos
    positivos onde o sinal de confiança não reflete o conteúdo real ou a
    veracidade da resposta.
    """

    def verify(self, supervisor_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Revalida a confiança booleana do Supervisor
        (try_answer_from_summaries). Ver docstring do módulo, item 1.

        Args:
            supervisor_result: dicionário com ao menos "answer" e "confident".

        Returns:
            Cópia do dicionário, com "confident"/"answer" possivelmente
            revogados e "verification_override": True se um padrão de
            incerteza foi detectado.
        """
        result = dict(supervisor_result)
        answer = result.get("answer", "") or ""
        confident = bool(result.get("confident", False))

        if not confident:
            return result

        matched_pattern = self._find_uncertainty_pattern(answer)
        if matched_pattern:
            logger.warning(
                "Verification Loop: resposta marcada como [CONFIANTE] pelo "
                "supervisor, mas contém padrão de incerteza ('{}'). "
                "Revogando confiança para forçar roteamento a especialista.",
                matched_pattern,
            )
            result["confident"] = False
            result["answer"] = ""
            result["verification_override"] = True

        return result

    def verify_specialist_confidence(
        self, answer: str, confidence: float, agent_name: str = ""
    ) -> Dict[str, Any]:
        """
        Revalida a confiança numérica de um Especialista (ou peer via A2A).
        Ver docstring do módulo, item 2.

        Args:
            answer: texto da resposta gerada pelo especialista.
            confidence: valor numérico de confiança calculado por
                SpecialistAgent.invoke() (heurística baseada em 3 frases fixas).
            agent_name: nome do especialista, usado apenas para log.

        Returns:
            {"confidence": float, "override": bool} — confidence ajustada
            para DEFAULT_LOW_CONFIDENCE se um padrão de incerteza for
            detectado no texto apesar de um valor numérico alto reportado.
        """
        matched_pattern = self._find_uncertainty_pattern(answer or "")

        if matched_pattern and confidence > DEFAULT_LOW_CONFIDENCE:
            logger.warning(
                "Verification Loop: especialista '{}' retornou confidence={:.2f}, "
                "mas a resposta contém padrão de incerteza ('{}'). "
                "Ajustando confidence para {:.2f}.",
                agent_name or "desconhecido",
                confidence,
                matched_pattern,
                DEFAULT_LOW_CONFIDENCE,
            )
            return {"confidence": DEFAULT_LOW_CONFIDENCE, "override": True}

        return {"confidence": confidence, "override": False}

    def verify_groundedness(
        self, answer: str, source_context: str, agent_name: str = ""
    ) -> Dict[str, Any]:
        """
        Detecta fabricação grosseira de dados de contato/identificação
        (e-mail, telefone, CNPJ) na resposta do Supervisor, quando esses
        dados não existem literalmente na fonte permitida (source_context).
        Ver docstring do módulo, item 3.

        NOTA DE ESCOPO: esta é uma checagem determinística e propositalmente
        limitada a padrões de contato. Não substitui uma avaliação de
        fidelidade semântica completa (ex.: listas de documentos plausíveis
        mas inventadas a partir de títulos de capítulo), que é escopo da
        suíte de regressão automatizada (Etapa 5) via métrica de
        faithfulness da biblioteca ragas.

        Args:
            answer: texto da resposta gerada pelo supervisor.
            source_context: texto da fonte que o supervisor tinha permissão
                de usar (ex.: SupervisorAgent.short_context_str).
            agent_name: nome do agente, usado apenas para log.

        Returns:
            {"grounded": bool, "override": bool, "fabricated_entities": list}
        """
        if not answer:
            return {"grounded": True, "override": False, "fabricated_entities": []}

        fabricated: List[Tuple[str, str]] = []
        for label, pattern in CONTACT_INFO_PATTERNS.items():
            for match in pattern.findall(answer):
                if match not in (source_context or ""):
                    fabricated.append((label, match))

        if fabricated:
            logger.warning(
                "Verification Loop: resposta de '{}' contém dado(s) de contato "
                "não presente(s) na fonte permitida (possível fabricação): {}. "
                "Revogando confiança.",
                agent_name or "desconhecido",
                fabricated,
            )
            return {"grounded": False, "override": True, "fabricated_entities": fabricated}

        return {"grounded": True, "override": False, "fabricated_entities": []}

    def _find_uncertainty_pattern(self, text: str) -> str:
        """Retorna o padrão (regex) de incerteza encontrado no texto, ou string vazia."""
        for pattern in _COMPILED_PATTERNS:
            if pattern.search(text):
                return pattern.pattern
        return ""