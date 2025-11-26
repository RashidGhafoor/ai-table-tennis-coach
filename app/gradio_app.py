"""
gradio_app.py

Simple Gradio UI that uploads a video, runs the vision->eval->coach pipeline, and shows results.
Run: python app/gradio_app.py
"""

import gradio as gr
import tempfile, os, json
from agents.vision_agent import analyze_video
from agents.eval_agent import score_shots
from agents.coach_agent import generate_coaching_plan

def analyze_file(video_path):
    # video_path is a temp path provided by gradio
    detections = analyze_video(video_path, max_frames=200, frame_stride=3)
    evaluations = score_shots(detections)
    plan = generate_coaching_plan(evaluations)
    out = {
        'detections_count': len(detections),
        'evaluations': evaluations,
        'plan': plan
    }
    # save report
    report_path = os.path.join(tempfile.gettempdir(), 'tt_report.json')
    with open(report_path, 'w') as f:
        json.dump(out, f, indent=2)
    return out

def main():
    iface = gr.Interface(
        fn=analyze_file,
        inputs=gr.Video(label="Upload Table-Tennis Clip (10-30s recommended)"),
        outputs=[gr.JSON(label="Analysis Report")],
        title="AI Table Tennis Coach (Demo)",
        description="Uploads a short table tennis clip and returns technique insights."
    )
    iface.launch(server_name='0.0.0.0', server_port=7860, share=False)

if __name__ == '__main__':
    main()
