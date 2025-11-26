"""
eval_agent.py

Takes vision agent detections and scores technique against simple rule-based heuristics.
Generates a structured evaluation report per shot/sequence.
"""

import numpy as np

def score_shots(detections, user_metadata=None):
    """Group detections into shots (very simple heuristic based on motion) and score each shot.
    Returns [{'shot_id': int, 'score': float, 'issues': [...], 'suggestions': [...]}, ...]
    """
    shots = []
    # naive grouping: every N frames -> a shot (for demo). Replace with actual shot detection logic.
    group_size = 10
    for i in range(0, len(detections), group_size):
        block = detections[i:i+group_size]
        avg_angle = np.nanmean([d['racket_angle'] or np.nan for d in block])
        issues = []
        suggestions = []
        score = 100.0
        if np.isnan(avg_angle):
            issues.append('Racket angle undetected in this sequence')
            score -= 30
        else:
            # heuristics: ideal angle depends on shot type; for demo assume 45-80 is good
            if avg_angle < 30 or avg_angle > 110:
                issues.append(f'Racket angle ({avg_angle:.1f}°) might be suboptimal')
                score -= 20
                suggestions.append('Experiment with a more open racket face around 45°–80° at contact')
        # posture heuristic (if keypoints exist)
        posture_issues = False
        for d in block:
            k = d.get('keypoints') or {}
            if k.get('left_elbow') and k.get('left_shoulder'):
                ex = k['left_elbow'][1]
                sx = k['left_shoulder'][1]
                if ex < sx - 30:  # elbow too high (y smaller is higher in image coords)
                    posture_issues = True
        if posture_issues:
            issues.append('Elbow appears high for some frames (may reduce control)')
            score -= 10
            suggestions.append('Work on shoulder-elbow-wrist alignment drills')
        shots.append({
            'shot_id': i//group_size,
            'score': max(score, 0),
            'issues': issues,
            'suggestions': suggestions,
            'avg_angle': float(avg_angle) if not np.isnan(avg_angle) else None,
            'frames': [d['frame_index'] for d in block]
        })
    return shots

if __name__ == '__main__':
    import json, sys
    data = json.load(open(sys.argv[1]))
    print(score_shots(data[:100]))
