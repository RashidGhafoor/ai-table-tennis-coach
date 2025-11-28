"""
insights_agent.py

LLM agent that iteratively refines diagnostic hypotheses before coaching.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from agents.gemini_utils import create_model, response_to_json

MAX_ATTEMPTS = 3

INSIGHTS_INSTRUCTIONS = """
You are a table-tennis diagnostics specialist. Produce JSON only—no markdown.
Schema:
{
  "hypothesis": "short paragraph",
  "evidence": ["bullet point", "..."],
  "confidence": 0.0
}
Rules:
- Evidence bullets must reference concrete stats or issues from tool_context/evaluations.
- confidence is a float between 0 and 1 (inclusive) with one decimal place.
- If you fail to follow the schema, the supervisor will resend your previous answer with a correction—fix it.
"""


def generate_insights(
    evaluations: List[Dict[str, Any]],
    *,
    tool_context: Dict[str, Any],
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run a bounded self-critique loop to produce consistent diagnostic insights.
    """
    user_profile = user_profile or {}
    base_payload = {
        "system_rules": INSIGHTS_INSTRUCTIONS.strip(),
        "user_profile": user_profile,
        "evaluation_samples": evaluations[:10],
        "tool_context": tool_context,
    }
    prompt = json.dumps(base_payload, indent=2)
    model = create_model()

    last_response: Optional[Dict[str, Any]] = None
    for attempt in range(MAX_ATTEMPTS):
        response = model.generate_content(prompt)
        parsed = response_to_json(response)
        ok, reason = _validate(parsed)
        if ok:
            parsed["llm_used"] = True
            return parsed
        # feed back the issue and try again
        feedback = {
            "feedback": f"Attempt {attempt + 1} was invalid: {reason}",
            "previous_response": parsed,
        }
        prompt = json.dumps({**base_payload, **feedback}, indent=2)
        last_response = parsed
    # If still invalid, return the last attempt for transparency
    final = last_response or {"hypothesis": "", "evidence": [], "confidence": 0.0}
    final["llm_used"] = True
    final["validation_error"] = "Exceeded retry budget while enforcing schema."
    return final


def _validate(payload: Dict[str, Any]) -> Tuple[bool, str]:
    hypothesis = payload.get("hypothesis")
    evidence = payload.get("evidence")
    confidence = payload.get("confidence")
    if not hypothesis or not isinstance(hypothesis, str):
        return False, "Missing hypothesis text."
    if not isinstance(evidence, list) or not evidence or not all(isinstance(e, str) for e in evidence):
        return False, "Evidence must be a non-empty list of strings."
    if not isinstance(confidence, (int, float)) or not (0 <= float(confidence) <= 1):
        return False, "Confidence must be between 0 and 1."
    return True, ""

