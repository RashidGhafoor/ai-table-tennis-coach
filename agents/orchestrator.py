"""
orchestrator.py

Coordinates the full multi-agent pipeline with observability hooks, caching,
and pause/resume semantics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from agents.vision_agent import analyze_video
from agents.eval_agent import score_shots
from agents.coach_agent import generate_coaching_plan
from services.session_service import InMemorySessionService, SessionRecord
from utils.observability import log_event, timed_span


class AgentOrchestrator:
    def __init__(
        self,
        session_service: InMemorySessionService,
        *,
        cache_root: Path | str = ".cache/runs",
    ) -> None:
        self.session_service = session_service
        self.cache_root = Path(cache_root)
        self.cache_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        video_path: str,
        session: SessionRecord,
        *,
        resume: bool = False,
    ) -> Dict[str, Any]:
        session_id = session.session_id
        cache_dir = self.cache_root / session_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        log_event("pipeline_start", session_id=session_id, resume=resume)

        detections = self._load_or_execute(
            cache_dir / "vision.json",
            resume,
            lambda: analyze_video(video_path, max_frames=200, frame_stride=3),
            stage="vision",
            session_id=session_id,
            summary_fields={"video_path": video_path},
        )
        evaluations = self._load_or_execute(
            cache_dir / "evaluation.json",
            resume,
            lambda: score_shots(detections),
            stage="evaluation",
            session_id=session_id,
            summary_fields={"detections": len(detections)},
        )
        with timed_span("coaching", session_id=session_id):
            plan = generate_coaching_plan(evaluations, user_profile=session.user_profile)
        log_event("coaching_complete", session_id=session_id, llm_used=plan.get("llm_used", False))

        result = {
            "session_id": session_id,
            "detections_count": len(detections),
            "evaluations": evaluations,
            "plan": plan,
            "resume_hint": self.session_service.get_resume_hint(session_id),
        }
        self.session_service.append_event(
            session_id,
            "pipeline_complete",
            {"frames": len(detections), "issues": [e.get("issues", []) for e in evaluations]},
        )
        self.session_service.set_last_result(session_id, result)
        log_event("pipeline_end", session_id=session_id)
        return result

    def _load_or_execute(
        self,
        cache_path: Path,
        resume: bool,
        fn,
        *,
        stage: str,
        session_id: str,
        summary_fields: Dict[str, Any] | None = None,
    ):
        summary_fields = summary_fields or {}
        if resume and cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            log_event("stage_resume", stage=stage, session_id=session_id, cache=str(cache_path))
            return data
        with timed_span(stage, session_id=session_id, **summary_fields):
            data = fn()
        cache_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.session_service.append_event(
            session_id,
            f"{stage}_complete",
            {"cache": str(cache_path), "summary": summary_fields},
        )
        log_event("stage_complete", stage=stage, session_id=session_id)
        return data

