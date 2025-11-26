# AI Table Tennis Coach Agent

This project is a Kaggle-friendly implementation of an AI Table Tennis Coach using the Google AI SDK (Gemini) and a multi-agent architecture.  
It includes:
- A Kaggle notebook (`notebooks/table_tennis_coach.ipynb`)
- Python modules for video/frame tools and agent orchestration
- A Gradio demo app to upload a video and get analysis
- Documentation and a step-by-step setup guide

**Important:** This repository provides starter code and a pipeline design. You must provide your Google Cloud API credentials and install dependencies before running on Kaggle.

## Contents
- `notebooks/table_tennis_coach.ipynb` — main Kaggle notebook
- `app/gradio_app.py` — Gradio demo app that runs the pipeline in notebook or local env
- `agents/vision_agent.py` — vision agent (frame extraction, pose estimation hooks)
- `agents/eval_agent.py` — technique evaluation & scoring
- `agents/coach_agent.py` — coaching agent producing drills and plan
- `tools/video_tools.py` — OpenCV helpers for frame extraction, angle computation
- `requirements.txt` — Python dependencies
- `README.md` — this file
- `SETUP.md` — detailed setup and step-by-step guide

## Quick Note
This code relies on:
- Google AI SDK / Gemini (Python package; install and configure)
- OpenCV (`opencv-python`)
- Mediapipe (or any pose estimator) — the code contains hooks to use mediapipe if available
- Gradio for demo UI

If you want a fully hosted Kaggle Notebook version, open `notebooks/table_tennis_coach.ipynb` in Kaggle, follow `SETUP.md`, and run cells sequentially.

