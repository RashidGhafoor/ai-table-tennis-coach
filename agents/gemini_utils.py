"""
gemini_utils.py

Shared helpers for configuring Gemini models and parsing responses.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

try:  # pragma: no cover
    import google.generativeai as genai
except ImportError:  # pragma: no cover
    genai = None

MODEL_ENV = "COACH_MODEL_NAME"
DEFAULT_MODEL = "gemini-2.5-flash-lite"


def _require_api_key() -> str:
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY is not set; cannot call Gemini.")
    return key


def create_model(*, tools: Optional[list] = None, function_call_mode: str = "NONE"):
    """
    Configure and return a GenerativeModel instance with consistent defaults.
    """
    if genai is None:
        raise RuntimeError("google-generativeai package is not installed.")
    key = _require_api_key()
    genai.configure(api_key=key)
    model_name = os.getenv(MODEL_ENV, DEFAULT_MODEL)
    return genai.GenerativeModel(
        model_name=model_name,
        tools=tools or [],
        tool_config={"function_calling_config": {"mode": function_call_mode}},
    )


def _strip_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1]
        if "```" in stripped:
            stripped = stripped.rsplit("```", 1)[0]
    return stripped.strip()


def extract_text(response: Any) -> str:
    """
    Collect textual parts from a Gemini response and stringify function calls for debugging.
    """
    parts: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        candidate_parts = []
        if content and hasattr(content, "parts"):
            candidate_parts = content.parts
        elif isinstance(content, dict):
            candidate_parts = content.get("parts", [])
        for part in candidate_parts:
            text = getattr(part, "text", None)
            if text is None and isinstance(part, dict):
                text = part.get("text")
            if text:
                parts.append(text)
                continue
            func_call = getattr(part, "function_call", None)
            if func_call is None and isinstance(part, dict):
                func_call = part.get("function_call")
            if func_call:
                func_payload = {
                    "name": getattr(func_call, "name", None) or getattr(func_call, "function_name", None),
                    "args": getattr(func_call, "args", None) or {},
                }
                parts.append(json.dumps({"function_call": func_payload}))
    if parts:
        return "\n".join(parts)
    if hasattr(response, "text") and response.text:
        return response.text
    return str(response)


def response_to_json(response: Any) -> Dict[str, Any]:
    """
    Convert a Gemini response into JSON, stripping markdown fences if present.
    """
    text = extract_text(response)
    text = _strip_fence(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Gemini response was not valid JSON: {text}") from exc

