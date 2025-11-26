# Setup Guide

This guide walks through the local-first workflow required for submission. Kaggle notebook instructions remain available at the end for reference.

---

## 1. Prerequisites
- Python 3.10+
- `pip` (or `uv`/`pipenv`)
- FFmpeg installed if you plan to preprocess/trim clips (optional but recommended)
- Google AI / Gemini API key (`google-generative-ai` Python package)

---

## 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Packages of note:
- `google-generative-ai` (Gemini SDK)
- `mediapipe`, `opencv-python`, `numpy`, `pandas`
- `gradio`, `tqdm`, `matplotlib`, `scikit-learn`

If the Google SDK name differs in your environment, install the package referenced in the [Vertex AI Agent Engine docs](https://cloud.google.com/vertex-ai/docs/agents).

---

## 3. Configure Credentials & Environment
```bash
export GOOGLE_API_KEY="YOUR_GEMINI_KEY"
# optional overrides
export COACH_MODEL_NAME="gemini-1.5-flash"
export AGENT_LOG_LEVEL=INFO
export SESSION_STORE_PATH=".cache/session_store.json"
```

You can also point `GOOGLE_APPLICATION_CREDENTIALS` to a service-account JSON if preferred.

---

## 4. Run the Local Gradio Demo
```bash
python app/gradio_app.py
```
Then open `http://127.0.0.1:7860`.

Inputs:
- **Video** (10–30s recommended)
- **Session ID (optional)** — paste a previous ID to resume cached stages
- **Skill Level / Goals** — passed into the memory service for personalization
- **Resume checkbox** — reuses cached detections/evaluations from `.cache/runs/<session_id>/`

Outputs:
- Structured JSON (detections, evaluations, coaching plan, tool context)
- Session ID string for subsequent runs

The session/memory service persists every run to `.cache/session_store.json` so you can pause after processing-heavy stages and resume later.

---

## 5. Observability & Logs
- Structured JSON logs (span start/end, stage completion, evaluation metrics) are emitted to stdout.
- Adjust verbosity with `AGENT_LOG_LEVEL`.
- Each run also writes cached artifacts to `.cache/runs/<session_id>/vision.json` & `evaluation.json`, which double as checkpoints for long-running operations.

---

## 6. Evaluation Harness
Use the CLI tool to compute precision/recall/F1 metrics.

```bash
# Synthetic dry-run (no video files needed)
python scripts/evaluate.py --mock

# Manifest run (provide your own dataset)
python scripts/evaluate.py --manifest data/eval_manifest.json
```

Manifest schema:
```json
[
  {
    "video": "/abs/path/clip.mp4",
    "expected_issues": ["Racket angle undetected in this sequence"],
    "user_profile": {"level": "Intermediate"}
  }
]
```

Reports are exported under `reports/`. Each execution also logs macro metrics via the observability layer.

---

## 7. Troubleshooting Tips
- **`import google.generativeai` fails** — ensure the package name in `requirements.txt` matches the SDK release available in your region.
- **Mediapipe not available** — substitute any keypoint detector that outputs wrist/shoulder joints and plug it into `vision_agent.py`.
- **Large videos stall** — trim clips to 10–30s or leverage the resume checkbox to reuse cached detections when tweaking parameters.
- **Sessions reset** — set `SESSION_STORE_PATH` to a persistent location or disable autosave in `InMemorySessionService` if running in ephemeral sandboxes.

