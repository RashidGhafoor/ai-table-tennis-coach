"""
coach_agent.py

LLM coach that produces drills and schedules from evaluation data.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from agents.gemini_utils import create_model, response_to_json

EVAL_LIMIT = 12

STRICT_INSTRUCTIONS = """
You are an elite table-tennis coach. Follow these rules exactly:
1. The response MUST be a single JSON object—no markdown, no conversation.
2. Required schema:
{
  "summary": "one concise paragraph",
  "drills": [
    {"name": "", "description": "", "focus": "", "repetitions": ""}
  ],
  "schedule": [
    {"day": 1, "focus": ""},
    {"day": 3, "focus": ""},
    {"day": 5, "focus": ""}
  ]
}
3. Every drill and schedule entry must reference insights or issues from tool_context/evaluations.
4. Use specific numbers/reps; avoid generic advice.
5. Violating the schema or adding extra text makes the answer invalid.
"""


def generate_coaching_plan(
    evaluations: List[Dict[str, Any]],
    *,
    tool_context: Dict[str, Any],
    insights: Optional[Dict[str, Any]] = None,
    user_profile: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    user_profile = user_profile or {}
    prompt = _build_prompt(evaluations, user_profile, tool_context, insights)
    model = create_model()
    response = model.generate_content(prompt)
    plan = response_to_json(response)
    plan["llm_used"] = True
    return plan


def _build_prompt(
    evaluations: List[Dict[str, Any]],
    user_profile: Dict[str, Any],
    tool_context: Dict[str, Any],
    insights: Optional[Dict[str, Any]],
) -> str:
    compact_evals = evaluations[:EVAL_LIMIT]
    payload = {
        "system_rules": STRICT_INSTRUCTIONS.strip(),
        "user_profile": user_profile,
        "evaluation_samples": compact_evals,
        "tool_context": tool_context,
        "insights_summary": insights or {},
    }
    return json.dumps(payload, indent=2)
