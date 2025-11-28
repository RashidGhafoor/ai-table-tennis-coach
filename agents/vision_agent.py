"""
vision_agent.py

Performs simple frame extraction and lightweight pose/racket detection.
Uses OpenCV and Mediapipe (if available). Also exposes a high-level analyze_video()
function that returns structured detections per frame.
"""

import cv2
import numpy as np
from tools.video_tools import extract_frames
try:
    import mediapipe as mp
    USE_MEDIAPIPE = True
except Exception:
    USE_MEDIAPIPE = False

def analyze_video(video_path, max_frames=300, frame_stride=3):
    """Extract frames and compute simple keypoints and racket angle proxies.
    Returns a list of dicts: [{'frame_index': int, 'keypoints': {...}, 'racket_angle': float, 'timestamp': float}, ...]
    """
    frames = extract_frames(video_path, max_frames=max_frames, frame_stride=frame_stride)
    results = []
    pose = None
    if USE_MEDIAPIPE:
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True)
    try:
        for i, (ts, img) in enumerate(frames):
            h, w = img.shape[:2]
            detection = {'frame_index': i, 'timestamp': ts, 'racket_angle': None, 'keypoints': {}}
            if pose:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = pose.process(img_rgb)
                if res.pose_landmarks:
                    landmarks = res.pose_landmarks.landmark
                    # extract a few landmarks by name indices if available
                    detection['keypoints'] = {
                        'left_shoulder': (landmarks[11].x * w, landmarks[11].y * h),
                        'right_shoulder': (landmarks[12].x * w, landmarks[12].y * h),
                        'left_elbow': (landmarks[13].x * w, landmarks[13].y * h),
                        'right_elbow': (landmarks[14].x * w, landmarks[14].y * h),
                        'left_wrist': (landmarks[15].x * w, landmarks[15].y * h),
                        'right_wrist': (landmarks[16].x * w, landmarks[16].y * h),
                        'left_hip': (landmarks[23].x * w, landmarks[23].y * h),
                        'right_hip': (landmarks[24].x * w, landmarks[24].y * h),
                    }
            # Racket angle proxy: detect largest edge direction in near-wrist area (simple heuristic)
            racket_angle = None
            try:
                racket_angle = compute_racket_angle(img, detection.get('keypoints', {}))
            except Exception:
                racket_angle = None
            detection['racket_angle'] = racket_angle
            results.append(detection)
    finally:
        if pose:
            pose.close()
    return results

def compute_racket_angle(img, keypoints):
    """Simple heuristic: look near the wrist for long linear contours and estimate angle.
    Returns angle in degrees (0-180).
    """
    from tools.video_tools import estimate_line_angle_near_point
    wrist = keypoints.get('right_wrist') or keypoints.get('left_wrist')
    if wrist is None:
        raise ValueError('No wrist keypoint provided')
    angle = estimate_line_angle_near_point(img, wrist, search_radius=120)
    return angle

if __name__ == '__main__':
    import sys
    video = sys.argv[1]
    res = analyze_video(video, max_frames=200)
    print('Frames analyzed:', len(res))
