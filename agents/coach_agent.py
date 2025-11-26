"""
coach_agent.py

LLM-enhanced coaching agent that transforms evaluation reports into actionable
drills and schedules. Falls back to a deterministic planner if the Google AI
SDK is unavailable.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

from agents.tool_registry import ToolRegistry

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - defensive import
    genai = None

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = os.getenv("COACH_MODEL_NAME", "gemini-1.5-flash")
CONTEXT_EVAL_LIMIT = 15  # lightweight context compaction


class LLMCoachAgent:
    """Coordinator that optionally leverages Gemini for coaching guidance."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        tool_registry: Optional[ToolRegistry] = None,
        enable_llm: bool = True,
    ) -> None:
        self.model_name = model_name
        self.tool_registry = tool_registry or ToolRegistry()
        self.enable_llm = enable_llm
        self._model = None
        self._configure_model()

    def _configure_model(self) -> None:
        if not (self.enable_llm and genai):
            LOGGER.warning("google-generative-ai not installed; using rule-based coach.")
            return
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            LOGGER.warning("GOOGLE_API_KEY not provided; defaulting to rule-based coach.")
            return
        try:
            genai.configure(api_key=api_key)
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                tools=self.tool_registry.schemas,
            )
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Failed to configure Gemini model: %s", exc, exc_info=True)
            self._model = None

    def generate_plan(
        self,
        evaluations: List[Dict[str, Any]],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        user_profile = user_profile or {}
        tool_context = self.tool_registry.gather_context(
            evaluations=evaluations, user_profile=user_profile
        )
        if not self._model:
            LOGGER.info("Using heuristic coaching plan.")
            plan = _generate_rule_based_plan(evaluations)
            plan["tool_context"] = tool_context
            plan["llm_used"] = False
            return plan
        prompt = self._build_prompt(evaluations, user_profile, tool_context)
        try:
            response = self._model.generate_content(prompt)
            plan = self._parse_response(response)
            plan["tool_context"] = tool_context
            plan["llm_used"] = True
            return plan
        except Exception as exc:  # pragma: no cover
            LOGGER.error("Gemini call failed; falling back to heuristic plan: %s", exc)
            plan = _generate_rule_based_plan(evaluations)
            plan["tool_context"] = tool_context
            plan["llm_used"] = False
            plan["llm_error"] = str(exc)
            return plan

    @staticmethod
    def _build_prompt(
        evaluations: List[Dict[str, Any]],
        user_profile: Dict[str, Any],
        tool_context: Dict[str, Any],
    ) -> str:
        compact_evals = evaluations[:CONTEXT_EVAL_LIMIT]
        prompt_payload = {
            "user_profile": user_profile,
            "evaluation_samples": compact_evals,
            "tool_context": tool_context,
            "instructions": (
                "Return JSON with keys: summary, drills (list of {name, description, focus, reps}), "
                "schedule (list of {day, focus}), and optional references array that cites tool names used."
            ),
        }
        return json.dumps(prompt_payload, indent=2)

    @staticmethod
    def _parse_response(response: Any) -> Dict[str, Any]:
        if hasattr(response, "text") and response.text:
            raw_text = response.text
        else:
            raw_text = str(response)
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # Fail-safe parsing by wrapping text into standard schema.
            return {
                "summary": raw_text[:500],
                "drills": [],
                "schedule": [],
                "references": ["llm-freeform-response"],
            }


def _generate_rule_based_plan(evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
    plan = {"summary": "", "drills": [], "schedule": []}
    problem_counts: Dict[str, int] = {}
    for evaluation in evaluations:
        for issue in evaluation.get("issues", []):
            problem_counts[issue] = problem_counts.get(issue, 0) + 1
    if not problem_counts:
        plan["summary"] = "Good technique overall. Focus on consistency and footwork."
    else:
        issues_sorted = sorted(problem_counts.items(), key=lambda pair: pair[1], reverse=True)
        plan["summary"] = "Common issues: " + ", ".join(
            [f"{issue} (x{count})" for issue, count in issues_sorted]
        )
    if any("Elbow" in issue or "elbow" in issue.lower() for issue in problem_counts):
        plan["drills"].append(
            {
                "name": "Elbow alignment drill",
                "description": (
                    "Practice slow-motion forehands focusing on keeping elbow below shoulder. "
                    "3 sets of 10 swings."
                ),
                "focus": "Upper-body structure",
            }
        )
    if any("Racket angle" in issue or "racket" in issue.lower() for issue in problem_counts):
        plan["drills"].append(
            {
                "name": "Open-face drill",
                "description": "Practice pushing the ball with a more open racket face. 5 sets of 15 feeds.",
                "focus": "Contact point",
            }
        )
    plan["schedule"] = [
        {"day": 1, "focus": f"Drills: {plan['drills'][0]['name']}" if plan["drills"] else "Consistency drills"},
        {"day": 3, "focus": "Multi-ball: work on rhythm and contact point"},
        {"day": 5, "focus": "Match play focusing on implementing technique"},
    ]
    return plan


def generate_coaching_plan(
    evaluations: List[Dict[str, Any]],
    user_profile: Optional[Dict[str, Any]] = None,
    *,
    prefer_llm: bool = True,
) -> Dict[str, Any]:
    agent = LLMCoachAgent(enable_llm=prefer_llm)
    return agent.generate_plan(evaluations, user_profile)


if __name__ == "__main__":  # pragma: no cover
    import sys

    with open(sys.argv[1], "r", encoding="utf-8") as fh:
        detections = json.load(fh)
    report = generate_coaching_plan(detections)
    print(json.dumps(report, indent=2))
