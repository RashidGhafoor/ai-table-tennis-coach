# AI Table Tennis Coach Agent

Table Tennis technique coach powered by a multi-agent pipeline:

1. **Vision Agent** — extracts frames, optional Mediapipe pose landmarks, racket-angle heuristics.
2. **Evaluation Agent** — scores sequences and surfaces issues via rule-based heuristics.
3. **LLM Insights Agent** — Gemini loop that self-critiques diagnostic hypotheses before coaching.
4. **LLM Coaching Agent** — strict-prompt Gemini planner that outputs drills + schedules.

---

## Key Features
- **Vision + Evaluation Agents** (`agents/vision_agent.py`, `agents/eval_agent.py`) for deterministic frame analysis and scoring.
- **LLM Insights Agent** (`agents/insights_agent.py`) that runs a self-critique loop to refine diagnostic hypotheses.
- **LLM Coaching Agent** (`agents/coach_agent.py`) with strict JSON-only prompting for drills and schedules.
- **Session & Memory Service** (`services/session_service.py`) for resumable runs stored under `.cache/`.
- **Observability Layer** (`utils/observability.py`) providing structured logging plus timing spans.
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
   export COACH_MODEL_NAME="gemini-2.5-flash-lite"
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
Run the automated evaluation:

```bash
# Synthetic dry-run (no video files needed)
python scripts/evaluate.py --mock

Reports land in `reports/evaluation_<timestamp>.json` with per-sample precision/recall/F1 and macro averages.

---

## Project Structure
- `app/gradio_app.py` — Gradio UI + pipeline invocation through the orchestrator
- `agents/vision_agent.py` — frame extraction & pose hooks
- `agents/eval_agent.py` — heuristic scoring engine
- `agents/insights_agent.py` — LLM diagnostics loop feeding the coach
- `agents/coach_agent.py` — strict JSON-only Gemini coach
- `agents/orchestrator.py` — orchestrates stages, caching, observability
- `agents/tool_registry.py` — MCP-style tool descriptors (`technique_breakdown`, `drill_lookup`)
- `services/session_service.py` — session + memory store with persistence
- `scripts/evaluate.py` — evaluation harness + metrics logging
- `tools/video_tools.py` — OpenCV helpers for frames & geometry
- `utils/observability.py` — logging/tracing utilities

---

See [`SETUP.md`](SETUP.md) for detailed environment instructions and troubleshooting tips.

