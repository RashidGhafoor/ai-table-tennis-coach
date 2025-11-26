"""
gradio_app.py

Simple Gradio UI that uploads a video, runs the vision->eval->coach pipeline, and shows results.
Run: python app/gradio_app.py
"""

import gradio as gr
import tempfile, os, json
from services.session_service import InMemorySessionService
from agents.orchestrator import AgentOrchestrator
from utils.observability import setup_logging

SESSION_SERVICE = InMemorySessionService()
setup_logging()
ORCHESTRATOR = AgentOrchestrator(SESSION_SERVICE)


def analyze_file(video_path, session_id="", skill_level="Intermediate", goals="", resume=False):
    """Gradio callback that runs the multi-agent pipeline and records session state."""
    session = SESSION_SERVICE.ensure_session(
        session_id.strip() or None,
        user_profile={"level": skill_level, "goals": goals},
    )
    out = ORCHESTRATOR.run(video_path, session, resume=resume)
    report_path = os.path.join(tempfile.gettempdir(), f"tt_report_{session.session_id}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out, session.session_id


def main():
    iface = gr.Interface(
        fn=analyze_file,
        inputs=[
            gr.Video(label="Upload Table-Tennis Clip (10-30s recommended)"),
            gr.Textbox(label="Session ID (optional)", placeholder="Leave blank to create a new session"),
            gr.Dropdown(
                choices=["Beginner", "Intermediate", "Advanced"],
                value="Intermediate",
                label="Skill Level",
            ),
            gr.Textbox(label="Goals / Notes", placeholder="e.g., Improve consistency against backspin"),
            gr.Checkbox(label="Resume from cached stages (if available)", value=False),
        ],
        outputs=[
            gr.JSON(label="Analysis Report"),
            gr.Textbox(label="Session ID (use to resume later)", interactive=False),
        ],
        title="AI Table Tennis Coach (Demo)",
        description="Uploads a short table tennis clip and returns technique insights.",
    )
    iface.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == '__main__':
    main()
