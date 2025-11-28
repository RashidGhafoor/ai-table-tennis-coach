"""
video_tools.py

Helper functions for frame extraction and simple geometric estimators.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

def extract_frames(video_path, max_frames=300, frame_stride=3):
    """Yield (timestamp_seconds, image_bgr) tuples for frames sampled from the video."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video path does not exist: {video_path}")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    idx = 0
    saved = 0
    while cap.isOpened() and saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % frame_stride == 0:
            ts = idx / fps
            frames.append((ts, frame.copy()))
            saved += 1
        idx += 1
    cap.release()
    return frames

def estimate_line_angle_near_point(img, point, search_radius=100):
    """Search a circular patch around `point` for dominant edge orientation and return angle in degrees."""
    x, y = int(point[0]), int(point[1])
    h, w = img.shape[:2]
    x0, y0 = max(0, x-search_radius), max(0, y-search_radius)
    x1, y1 = min(w, x+search_radius), min(h, y+search_radius)
    patch = img[y0:y1, x0:x1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=30, maxLineGap=10)
    if lines is None:
        return None
    # choose the longest line
    longest = max(lines, key=lambda l: np.hypot(l[0][2]-l[0][0], l[0][3]-l[0][1]))
    x1, y1, x2, y2 = longest[0]
    angle_rad = np.arctan2((y2-y1), (x2-x1))
    angle_deg = np.degrees(angle_rad)
    # normalize to 0-180
    angle_deg = abs(angle_deg)
    if angle_deg > 180:
        angle_deg = angle_deg % 180
    return angle_deg

def compute_angle(a, b, c):
    """Return angle ABC in degrees given three points a,b,c (x,y)."""
    a = np.array(a); b = np.array(b); c = np.array(c)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    return ang
