"""
tool_registry.py

Defines lightweight MCP-style tool descriptors that wrap the vision/evaluation
logic so that higher-level agents (LLM coach, orchestrator, etc.) can reason
over structured insights and domain knowledge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import statistics

try:  # pragma: no cover - optional dependency is only needed for Gemini tool specs
    from google.genai import types as genai_types
except Exception:  # pragma: no cover
    genai_types = None


class BaseTool:
    """Minimal tool interface inspired by MCP tool contracts."""

    name: str = ""
    description: str = ""

    def run(self, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def _parameters_schema(self):  # pragma: no cover - subclasses override
        raise NotImplementedError

    def function_declaration(self):
        if genai_types is None:
            return None
        parameters = self._parameters_schema()
        if parameters is None:
            return None
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=parameters,
        )


class TechniqueBreakdownTool(BaseTool):
    name = "technique_breakdown"
    description = (
        "Aggregates evaluation outputs to highlight scoring trends, "
        "dominant issues, and consistency metrics."
    )

    def run(self, *, evaluations: Optional[List[Dict[str, Any]]] = None, **_) -> Dict[str, Any]:
        evaluations = evaluations or []
        scores = [e.get("score") for e in evaluations if isinstance(e.get("score"), (int, float))]
        score_summary = {
            "count": len(scores),
            "average": round(statistics.mean(scores), 2) if scores else None,
            "stdev": round(statistics.pstdev(scores), 2) if len(scores) > 1 else None,
            "best": max(scores) if scores else None,
            "worst": min(scores) if scores else None,
        }
        issue_counts: Dict[str, int] = {}
        for evaluation in evaluations:
            for issue in evaluation.get("issues", []):
                issue_counts[issue] = issue_counts.get(issue, 0) + 1
        top_issues = sorted(issue_counts.items(), key=lambda kv: kv[1], reverse=True)
        return {
            "score_summary": score_summary,
            "top_issues": [{"issue": issue, "count": count} for issue, count in top_issues],
            "frames_analyzed": sum(len(e.get("frames", [])) for e in evaluations),
        }

    def _parameters_schema(self):
        if genai_types is None:
            return None
        return genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "evaluations": genai_types.Schema(
                    type=genai_types.Type.ARRAY,
                    description="List of evaluation dicts with score, issues, and suggestions.",
                    items=genai_types.Schema(type=genai_types.Type.OBJECT),
                )
            },
            required=["evaluations"],
        )


class DrillLookupTool(BaseTool):
    name = "drill_lookup"
    description = "Returns targeted drills pulled from an embedded knowledge base for a given issue."

    _DRILL_LIBRARY = [
        {
            "keywords": ["racket angle", "open face", "contact point"],
            "name": "Open-Face Progression",
            "description": "Feed multi-ball pushes focusing on keeping the racket between 45°-80°.",
            "repetitions": "5 sets x 15 balls",
        },
        {
            "keywords": ["elbow", "alignment", "posture"],
            "name": "Elbow Ladder Drill",
            "description": "Shadow-swing forehands in front of a mirror keeping elbow below shoulder.",
            "repetitions": "3 sets x 12 swings",
        },
        {
            "keywords": ["footwork", "timing", "rhythm"],
            "name": "Two-Point Footwork",
            "description": "FH from BH corner alternating wide/outside placements with quick recovery.",
            "repetitions": "6 sets x 10 balls",
        },
    ]

    def run(
        self,
        *,
        issue: str,
        skill_level: Optional[str] = None,
        **_,
    ) -> Dict[str, Any]:
        skill_level = (skill_level or "intermediate").lower()
        issue_lower = issue.lower()
        matches = [
            drill
            for drill in self._DRILL_LIBRARY
            if any(keyword in issue_lower for keyword in drill["keywords"])
        ]
        if not matches:
            matches = [
                {
                    "name": "Consistency Loop",
                    "description": "Alternate FH/BH control shots aiming for high rally count.",
                    "repetitions": "10 minutes continuous",
                }
            ]
        return {"issue": issue, "skill_level": skill_level, "drills": matches}

    def _parameters_schema(self):
        if genai_types is None:
            return None
        return genai_types.Schema(
            type=genai_types.Type.OBJECT,
            properties={
                "issue": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="Issue to search drills for.",
                ),
                "skill_level": genai_types.Schema(
                    type=genai_types.Type.STRING,
                    description="Optional user skill level for personalization.",
                ),
            },
            required=["issue"],
        )


@dataclass
class ToolRegistry:
    tools: List[BaseTool] = field(default_factory=list)

    def __post_init__(self):
        if not self.tools:
            self.tools = [
                TechniqueBreakdownTool(),
                DrillLookupTool(),
            ]

    @property
    def schemas(self) -> List[Any]:
        """Return google.genai Tool definitions for Gemini."""
        if genai_types is None:
            return []
        declarations = []
        for tool in self.tools:
            decl = tool.function_declaration()
            if decl:
                declarations.append(decl)
        return [genai_types.Tool(function_declarations=[decl]) for decl in declarations]

    def invoke(self, name: str, **kwargs) -> Dict[str, Any]:
        for tool in self.tools:
            if tool.name == name:
                return tool.run(**kwargs)
        raise ValueError(f"Tool '{name}' not registered.")

    def gather_context(
        self,
        *,
        evaluations: Optional[List[Dict[str, Any]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke tools with shared context to build structured evidence for the LLM agent."""
        evaluations = evaluations or []
        user_profile = user_profile or {}
        issue_keywords = self._collect_issues(evaluations)
        context: Dict[str, Any] = {}
        for tool in self.tools:
            tool_kwargs = {
                "evaluations": evaluations,
                "issues": issue_keywords,
                "user_profile": user_profile,
            }
            if isinstance(tool, DrillLookupTool):
                # Run once per top issue to mimic tool calls and collect references.
                drill_entries = []
                for issue in issue_keywords[:3] or ["overall technique"]:
                    drill_entries.append(tool.run(issue=issue, skill_level=user_profile.get("level")))
                context[tool.name] = drill_entries
            else:
                context[tool.name] = tool.run(evaluations=evaluations, user_profile=user_profile)
        return context

    @staticmethod
    def _collect_issues(evaluations: List[Dict[str, Any]]) -> List[str]:
        issues: List[str] = []
        for evaluation in evaluations:
            issues.extend(evaluation.get("issues", []))
        # de-duplicate while preserving order
        seen = set()
        unique = []
        for issue in issues:
            if issue not in seen:
                seen.add(issue)
                unique.append(issue)
        return unique

