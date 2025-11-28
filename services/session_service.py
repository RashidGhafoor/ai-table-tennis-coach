"""
session_service.py

Provides a lightweight in-memory session + memory service so that agent runs
can persist user context, past evaluations, and partial progress. The service
stores state to disk (JSON) to support pause/resume semantics required by the
course rubric.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_STORE = Path(os.getenv("SESSION_STORE_PATH", ".cache/session_store.json"))

@dataclass
class SessionRecord:
    session_id: str
    user_profile: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    last_result: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

class InMemorySessionService:
    def __init__(self, *, store_path: Path = DEFAULT_STORE, autosave: bool = True) -> None:
        self.store_path = store_path
        self.autosave = autosave
        self.sessions: Dict[str, SessionRecord] = {}
        self._load_from_disk()

    def create_session(self, user_profile: Optional[Dict[str, Any]] = None) -> SessionRecord:
        session_id = uuid.uuid4().hex[:12]
        record = SessionRecord(session_id=session_id, user_profile=user_profile or {})
        self.sessions[session_id] = record
        self._persist()
        return record

    def ensure_session(
        self,
        session_id: Optional[str],
        user_profile: Optional[Dict[str, Any]] = None,
    ) -> SessionRecord:
        if session_id and session_id in self.sessions:
            record = self.sessions[session_id]
            if user_profile:
                record.user_profile.update({k: v for k, v in user_profile.items() if v})
                record.updated_at = time.time()
                self._persist()
            return record
        return self.create_session(user_profile=user_profile)

    def get(self, session_id: str) -> Optional[SessionRecord]:
        return self.sessions.get(session_id)

    def append_event(self, session_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        record = self.sessions.setdefault(session_id, SessionRecord(session_id=session_id))
        record.events.append(
            {
                "type": event_type,
                "ts": time.time(),
                "payload": payload,
            }
        )
        record.updated_at = time.time()
        self._persist()

    def set_last_result(self, session_id: str, result: Dict[str, Any]) -> None:
        record = self.sessions.setdefault(session_id, SessionRecord(session_id=session_id))
        record.last_result = result
        record.updated_at = time.time()
        self._persist()

    def get_resume_hint(self, session_id: str) -> Optional[str]:
        record = self.sessions.get(session_id)
        if not record or not record.events:
            return None
        last_event = record.events[-1]
        return f"Last event '{last_event['type']}' at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_event['ts']))}"

    def dump_sessions(self) -> Dict[str, Any]:
        return {
            session_id: {
                "user_profile": record.user_profile,
                "events": record.events,
                "last_result": record.last_result,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }
            for session_id, record in self.sessions.items()
        }

    def _load_from_disk(self) -> None:
        if not self.store_path.exists():
            return
        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            for session_id, payload in data.items():
                self.sessions[session_id] = SessionRecord(
                    session_id=session_id,
                    user_profile=payload.get("user_profile", {}),
                    events=payload.get("events", []),
                    last_result=payload.get("last_result"),
                    created_at=payload.get("created_at", time.time()),
                    updated_at=payload.get("updated_at", time.time()),
                )
        except Exception:
            # On corruption, start fresh but keep file for debugging.
            backup = self.store_path.with_suffix(".corrupt")
            self.store_path.replace(backup)
            self.sessions = {}

    def _persist(self) -> None:
        if not self.autosave:
            return
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            session_id: {
                "user_profile": record.user_profile,
                "events": record.events,
                "last_result": record.last_result,
                "created_at": record.created_at,
                "updated_at": record.updated_at,
            }
            for session_id, record in self.sessions.items()
        }
        self.store_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

