# AI Table Tennis Coach Agent

Table Tennis technique coach powered by a multi-agent pipeline:

1. **Vision Agent** — extracts frames, optional Mediapipe pose landmarks, racket-angle heuristics.
2. **Evaluation Agent** — scores sequences and surfaces issues via rule-based heuristics.
3. **LLM Coaching Agent** — Gemini-driven planner that calls MCP-style tools to craft drills, schedules, and references.

The repo now targets a local-first submission (no notebook required) while keeping the Kaggle notebook as an optional entrypoint.

---

## Key Features
- **LLM Coaching Agent** (`agents/coach_agent.py`) powered by Google AI SDK with structured prompts, context compaction, and MCP-style tool schemas.
- **Tool Registry** (`agents/tool_registry.py`) exposing `technique_breakdown` and `drill_lookup` tools, reusable by any orchestrated agent.
- **Session & Memory Service** (`services/session_service.py`) for stateful runs, user profiles, and pause/resume metadata persisted to `.cache/session_store.json`.
- **Observability Layer** (`utils/observability.py`) providing structured logging plus timing spans; integrated through the new `AgentOrchestrator`.
- **Agent Orchestrator** (`agents/orchestrator.py`) coordinating stages, caching outputs, and enabling resume-by-stage semantics.
- **Evaluation Harness** (`scripts/evaluate.py`) measuring precision/recall on issue detection (manifest or synthetic mode) and writing reports to `reports/`.

---

## Quick Start (Local)
1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure environment**
   ```bash
   export GOOGLE_API_KEY="YOUR_GEMINI_KEY"
   # optional
   export COACH_MODEL_NAME="gemini-1.5-flash"
   export AGENT_LOG_LEVEL=INFO
   ```
3. **Run the Gradio demo**
   ```bash
   python app/gradio_app.py
   ```
   - Upload a 10–30s clip
   - (Optional) provide an existing Session ID to resume cached stages
   - Adjust skill level + goals to personalize the coaching agent

---

## Sessions, Memory & Resume
- Every run is stored in `.cache/session_store.json` with events, metadata, and last results.
- The orchestrator caches intermediate outputs under `.cache/runs/<session_id>/`.
- In the Gradio UI you can:
  - Leave the Session ID blank to create a new session
  - Paste a previous Session ID and check **Resume from cached stages** to skip already completed phases
- Programmatic access is available via `InMemorySessionService`, enabling long-term memory or integration with other storage layers if desired.

---

## Observability
- Logging is centralized through `utils/observability.py` and enabled automatically in the Gradio app / evaluation script.
- Events such as `span_start`, `stage_complete`, `coaching_complete`, and evaluation metrics are emitted as JSON strings—pipe them to any log collector.
- Timed spans provide latency metrics for each stage; adjust verbosity via `AGENT_LOG_LEVEL`.

---

## Evaluation Harness
Run the automated evaluation (manifest or mock mode):

```bash
# Synthetic dry-run (no video files needed)
python scripts/evaluate.py --mock

# Manifest mode
python scripts/evaluate.py --manifest data/eval_manifest.json
```

Each manifest entry should look like:

```json
{
  "video": "/path/to/clip.mp4",
  "expected_issues": ["Racket angle undetected in this sequence"],
  "user_profile": {"level": "Intermediate"}
}
```

Reports land in `reports/evaluation_<timestamp>.json` with per-sample precision/recall/F1 and macro averages.

---

## Project Structure
- `app/gradio_app.py` — Gradio UI + pipeline invocation through the orchestrator
- `agents/vision_agent.py` — frame extraction & pose hooks
- `agents/eval_agent.py` — heuristic scoring engine
- `agents/coach_agent.py` — Gemini-coach with tool integration & fallback heuristics
- `agents/orchestrator.py` — orchestrates stages, caching, observability
- `agents/tool_registry.py` — MCP-style tool descriptors (`technique_breakdown`, `drill_lookup`)
- `services/session_service.py` — session + memory store with persistence
- `scripts/evaluate.py` — evaluation harness + metrics logging
- `tools/video_tools.py` — OpenCV helpers for frames & geometry
- `utils/observability.py` — logging/tracing utilities

---

## Course Requirement Checklist
| Requirement | Implementation |
| --- | --- |
| Multi-agent system | Vision → Evaluation → LLM Coaching agents orchestrated sequentially with caching |
| Tools (MCP/custom) | `agents/tool_registry.py` exposes tools consumed by the LLM agent |
| Sessions & Memory | `InMemorySessionService` + resume support in Gradio/orchestrator |
| Long-running operations | Stage-level caching + resume flag to skip expensive recomputation |
| Observability | Structured logging & timed spans in `utils/observability.py` + orchestration events |
| Agent evaluation | `scripts/evaluate.py` outputs precision/recall/F1 reports |
| Deployment | Local Gradio app + CLI scripts ready for GitHub submission (no notebook required) |

See [`SETUP.md`](SETUP.md) for detailed environment instructions and troubleshooting tips.

