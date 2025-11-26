"""
observability.py

Centralized logging + tracing helpers to provide observability over the agent
pipeline without requiring external services.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Optional


def setup_logging(default_level: int = logging.INFO) -> None:
    level = os.getenv("AGENT_LOG_LEVEL", default_level)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def log_event(event_type: str, **fields: Any) -> None:
    payload = {"event": event_type, **fields}
    logging.getLogger("agent").info(json.dumps(payload))


@contextmanager
def timed_span(span: str, session_id: Optional[str] = None, **fields: Any):
    trace_id = uuid.uuid4().hex[:8]
    start = time.time()
    log_event("span_start", span=span, trace_id=trace_id, session_id=session_id, **fields)
    try:
        yield
    finally:
        duration = round((time.time() - start) * 1000, 2)
        log_event(
            "span_end",
            span=span,
            trace_id=trace_id,
            duration_ms=duration,
            session_id=session_id,
            **fields,
        )

