# Setup Guide (Detailed Step-by-Step)

This guide explains how to set up and run the AI Table Tennis Coach inside a Kaggle Notebook or locally.

## 1) Get Google AI / Gemini API access
1. Create a Google Cloud project and enable the Generative AI API (Gemini) or install the Google AI SDK per Google's documentation.
2. Create an API key or configure Application Default Credentials.
3. Save your API key securely. In Kaggle, set it as an environment variable (see below).

## 2) Open the Kaggle Notebook
1. Upload the `notebooks/table_tennis_coach.ipynb` file into a new Kaggle Notebook (Notebook -> Add or upload file).
2. Also upload `agents/`, `tools/`, and `app/` folders if you prefer to run the modular code.

## 3) Prepare the Kaggle environment
In a Kaggle notebook cell, run:
```
!pip install -r /kaggle/input/your-uploaded-requirements/requirements.txt
# or if you placed requirements in the notebook working directory:
!pip install -r requirements.txt
```

### Required packages (see `requirements.txt`)
- google-ai-sdk (name may vary; replace with `google-generative-ai` / `google-api-core` if required)
- mediapipe
- opencv-python
- numpy
- pandas
- gradio
- matplotlib
- tqdm

If you hit import errors for the Google SDK, check the official Google Generative AI Python package name and install accordingly:
- Example: `pip install google-generative-ai` or `pip install google-api-python-client`

## 4) Configure credentials
Set environment variable (Kaggle notebook cell):
```python
import os
os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"
# or use service account JSON and set GOOGLE_APPLICATION_CREDENTIALS
```

## 5) Run the notebook
- Run cells in order. The notebook will:
  - load helper modules
  - extract frames from an uploaded video
  - run pose estimation (mediapipe by default)
  - call the Vision Agent (Gemini) for higher-level analysis
  - run Technique Evaluation Agent and Coaching Agent
  - display results and provide a downloadable report

## 6) Optional: Run Gradio demo locally (or in notebook)
To run the Gradio app locally:
```
python app/gradio_app.py
```
Then open `http://127.0.0.1:7860`.

## 7) Troubleshooting & Tips
- If Mediapipe isn't available in Kaggle's environment, you can use any pose estimator that outputs keypoints (OpenPose, BlazePose, etc.)
- The Google AI SDK import path might change — check Google docs if `import google.generativeai` fails.
- Video processing can be slow on large videos. Trim to short clips (10–30 seconds) for quicker feedback.

## 8) Evaluation
- The notebook includes an evaluation section with labeled samples. Provide a few manually annotated clips to measure detection/score accuracy.

