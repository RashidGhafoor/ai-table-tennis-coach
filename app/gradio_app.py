"""
gradio_app.py

Local Gradio UI for the AI Table Tennis Coach workflow.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List

import gradio as gr
import pandas as pd

from agents.orchestrator import AgentOrchestrator
from services.session_service import InMemorySessionService
from utils.observability import setup_logging

BASIC_UI_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto;
    padding: 24px 24px 48px !important;
}

.section-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 16px;
    gap: 12px;
    background: var(--panel-background-fill);
}

.section-card h3 {
    margin-top: 0;
}

.stage-progress {
    margin-bottom: 16px;
}

footer {
    display: none !important;
}
"""

SESSION_SERVICE = InMemorySessionService()
setup_logging()
ORCHESTRATOR = AgentOrchestrator(SESSION_SERVICE)


def _format_evidence(items: List[str]) -> str:
    if not items:
        return "_No specific evidence provided._"
    return "\n".join(f"- {text}" for text in items)

def _build_dataframe(rows: List[Dict[str, Any]], columns: List[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows)
    missing = [col for col in columns if col not in df.columns]
    for col in missing:
        df[col] = ""
    return df[columns]

def analyze_file(video_path, session_id="", skill_level="Intermediate", goals="", resume=False):
    """Gradio callback that runs the multi-agent pipeline and returns structured output."""
    session = SESSION_SERVICE.ensure_session(
        session_id.strip() or None,
        user_profile={"level": skill_level, "goals": goals},
    )
    out = ORCHESTRATOR.run(video_path, session, resume=resume)
    report_path = os.path.join(tempfile.gettempdir(), f"tt_report_{session.session_id}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    evaluations = out.get("evaluations", [])
    scores = [e.get("score") for e in evaluations if isinstance(e.get("score"), (int, float))]
    avg_score = round(sum(scores) / len(scores), 1) if scores else "N/A"
    issue_count = sum(len(e.get("issues", [])) for e in evaluations)
    detection_count = out.get("detections_count") or len(evaluations) * 10

    stats_md = (
        f"**Detections:** {detection_count}\n\n"
        f"- Average score: {avg_score}\n"
        f"- Issues flagged: {issue_count}\n"
        f"- Shots analyzed: {len(evaluations)}"
    )

    diagnostics = out.get("diagnostics") or {}
    diag_hypothesis = diagnostics.get("hypothesis") or "No diagnostics available."
    diag_evidence = _format_evidence(diagnostics.get("evidence") or [])
    diag_confidence = diagnostics.get("confidence")
    diag_confidence_text = f"{diag_confidence * 100:.0f}%" if isinstance(diag_confidence, (int, float)) else "N/A"

    plan = out.get("plan") or {}
    summary_text = plan.get("summary") or "No coaching summary generated."
    drills = plan.get("drills") or []
    schedule = plan.get("schedule") or []

    drills_df = _build_dataframe(
        drills,
        ["name", "description", "focus", "repetitions"],
    )
    schedule_df = _build_dataframe(
        schedule,
        ["day", "focus"],
    )

    stage_rows = [
        {"Stage": "Vision", "Status": "Complete" if detection_count else "Pending"},
        {"Stage": "Evaluation", "Status": "Complete" if evaluations else "Pending"},
        {"Stage": "Insights", "Status": "Complete" if diagnostics else "Pending"},
        {"Stage": "Coaching", "Status": "Complete" if plan else "Pending"},
    ]
    stage_status = pd.DataFrame(stage_rows)

    resume_hint = out.get("resume_hint", "")

    return (
        summary_text,
        diag_hypothesis,
        diag_evidence,
        diag_confidence_text,
        drills_df,
        schedule_df,
        stats_md,
        stage_status,
        session.session_id,
        resume_hint,
        report_path,
    )


def main():
    with gr.Blocks(title="AI Table Tennis Coach") as demo:
        gr.HTML(f"<style>{BASIC_UI_CSS}</style>")
        gr.Markdown(
            "# AI Table Tennis Coach\n"
            "Upload a 10-30 second clip to get diagnostics, coaching insights, and drills."
        )

        with gr.Row():
            with gr.Column(scale=40):
                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### Upload & Context")
                    video_input = gr.Video(label="Match Rally Clip")
                    session_input = gr.Textbox(
                        label="Session ID",
                        placeholder="Leave blank to create a new session",
                    )
                    level_input = gr.Dropdown(
                        choices=["Beginner", "Intermediate", "Advanced"],
                        value="Intermediate",
                        label="Skill Level",
                    )
                    goals_input = gr.Textbox(
                        label="Goals / Notes",
                        placeholder="e.g., Improve consistency against backspin",
                    )
                    resume_input = gr.Checkbox(label="Resume from cached stages (if available)", value=False)
                    submit_btn = gr.Button("Analyze Clip", variant="primary")

            with gr.Column(scale=60):
                stage_table = gr.DataFrame(label="Tools Used", interactive=False, elem_classes="section-card stage-progress")

                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### Coaching Summary")
                    summary_box = gr.Textbox(
                        label="Coaching Summary",
                        lines=6,
                        interactive=False,
                    )
                    stats_md = gr.Markdown(label="Session Stats")

                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### Diagnostics")
                    diagnostics_conf = gr.Textbox(
                        label="Diagnostics Confidence",
                        interactive=False,
                    )
                    diagnostics_hypothesis = gr.Textbox(
                        label="Diagnostic Hypothesis",
                        lines=4,
                        interactive=False,
                    )
                    diagnostics_evidence = gr.Markdown(label="Supporting Evidence")

                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### Practice Plan")
                    drills_table = gr.DataFrame(label="Drills", interactive=False)
                    schedule_table = gr.DataFrame(label="Schedule", interactive=False)

                with gr.Group(elem_classes="section-card"):
                    gr.Markdown("### Session Tools")
                    session_id_out = gr.Textbox(label="Session ID", interactive=False)
                    resume_hint_out = gr.Textbox(label="Resume Hint", interactive=False)
                    report_file = gr.File(label="Download Full Report", interactive=False)

        submit_btn.click(
            fn=analyze_file,
            inputs=[video_input, session_input, level_input, goals_input, resume_input],
            outputs=[
                summary_box,
                diagnostics_hypothesis,
                diagnostics_evidence,
                diagnostics_conf,
                drills_table,
                schedule_table,
                stats_md,
                stage_table,
                session_id_out,
                resume_hint_out,
                report_file,
            ],
        )

        gr.ClearButton(
            components=[
                video_input,
                session_input,
                level_input,
                goals_input,
                resume_input,
                summary_box,
                diagnostics_hypothesis,
                diagnostics_evidence,
                diagnostics_conf,
                drills_table,
                schedule_table,
                stats_md,
                stage_table,
                session_id_out,
                resume_hint_out,
                report_file,
            ]
        )

        demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

if __name__ == "__main__":
    main()
