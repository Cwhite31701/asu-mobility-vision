"""
Detection and overlay utilities for the YOLOv8 mobility detector.
"""

import cv2
import numpy as np
from typing import Optional


# Class display config
CLASS_CONFIG = {
    0: {  # person → Walker
        "label": "Walker",
        "color": (64, 220, 160),   # mint green
        "icon": "🚶",
    },
    1: {  # bike_wheel → Wheeled
        "label": "Wheeled",
        "color": (255, 160, 30),   # amber orange
        "icon": "🚲",
    },
}

FONT        = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE  = 0.55
THICKNESS   = 2


def draw_detections(
    frame: np.ndarray,
    detections: list[dict],
    show_conf: bool = True,
    show_track_id: bool = True,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on a frame.

    Args:
        frame: BGR frame
        detections: list of dicts with x1,y1,x2,y2,cls_id,conf (and optionally track_id)

    Returns:
        Annotated frame (copy)
    """
    out = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        cls_id = int(det.get("cls_id", 0))
        conf   = det.get("conf", 1.0)
        track_id = det.get("track_id", None)

        cfg   = CLASS_CONFIG.get(cls_id, CLASS_CONFIG[0])
        color = cfg["color"]
        label = cfg["label"]

        # Draw filled semi-transparent box edges
        overlay = out.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.08, out, 0.92, 0, out)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, THICKNESS)

        # Build label string
        tag_parts = [label]
        if show_conf:
            tag_parts.append(f"{conf:.0%}")
        if show_track_id and track_id is not None:
            tag_parts.append(f"#{track_id}")
        tag = " ".join(tag_parts)

        # Label background
        (tw, th), _ = cv2.getTextSize(tag, FONT, FONT_SCALE, THICKNESS)
        lbl_y = max(y1 - 4, th + 4)
        cv2.rectangle(out, (x1, lbl_y - th - 4), (x1 + tw + 6, lbl_y + 2), color, -1)
        cv2.putText(out, tag, (x1 + 3, lbl_y - 1), FONT, FONT_SCALE, (10, 10, 10), THICKNESS)

    return out


def draw_hud(
    frame: np.ndarray,
    walkers: int,
    wheeled: int,
    score: float,
    status: str,
    source_label: str = "LIVE",
    rolling_avg: float = 0.0,
) -> np.ndarray:
    """
    Draw HUD overlay (counts, score, status) in top-left corner.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # HUD background panel
    panel_w, panel_h = 280, 130
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (10, 10, 20), -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    STATUS_COLORS = {
        "LOW":    (64, 220, 100),
        "MEDIUM": (255, 200, 30),
        "HIGH":   (50, 50, 255),
    }
    sc = STATUS_COLORS.get(status, (200, 200, 200))

    # Status indicator dot + text
    cv2.circle(out, (18, 18), 8, sc, -1)
    cv2.putText(out, f"{status} CONGESTION", (32, 24), FONT, 0.52, sc, 1)

    # Walker / Wheeled counts
    cv2.putText(out, f"Walkers:  {walkers}", (12, 50), FONT, 0.5, CLASS_CONFIG[0]["color"], 1)
    cv2.putText(out, f"Wheeled:  {wheeled}", (12, 70), FONT, 0.5, CLASS_CONFIG[1]["color"], 1)
    cv2.putText(out, f"Score:    {score:.1f}", (12, 90),  FONT, 0.5, (200, 200, 255), 1)
    cv2.putText(out, f"Rolling:  {rolling_avg:.1f}",  (12, 110), FONT, 0.5, (160, 160, 200), 1)

    # Source badge (top-right)
    badge = f"  {source_label}  "
    (bw, bh), _ = cv2.getTextSize(badge, FONT, 0.45, 1)
    bx = w - bw - 10
    cv2.rectangle(out, (bx - 4, 6), (bx + bw + 2, 6 + bh + 6), (0, 0, 180), -1)
    cv2.putText(out, badge, (bx, 6 + bh + 2), FONT, 0.45, (255, 255, 255), 1)

    return out


def build_heatmap(
    frame: np.ndarray,
    detection_history: list[list[dict]],
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Overlay a movement density heatmap from recent detection history.

    Args:
        frame: current BGR frame
        detection_history: last N frames' detection lists
        alpha: heatmap blend strength

    Returns:
        Frame with heatmap overlay
    """
    h, w = frame.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for frame_dets in detection_history:
        for det in frame_dets:
            cx = int((det["x1"] + det["x2"]) / 2)
            cy = int((det["y1"] + det["y2"]) / 2)
            cx = max(0, min(w - 1, cx))
            cy = max(0, min(h - 1, cy))
            bw = max(1, int(det["x2"] - det["x1"]))
            bh = max(1, int(det["y2"] - det["y1"]))
            sigma = max(bw, bh) // 2
            # Gaussian splat
            y0, y1 = max(0, cy - sigma), min(h, cy + sigma)
            x0, x1 = max(0, cx - sigma), min(w, cx + sigma)
            if y1 > y0 and x1 > x0:
                patch = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
                for py in range(y1 - y0):
                    for px in range(x1 - x0):
                        dy = py + y0 - cy
                        dx = px + x0 - cx
                        patch[py, px] = np.exp(-(dy**2 + dx**2) / (2 * max(sigma**2, 1)))
                heatmap[y0:y1, x0:x1] += patch

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    mask = heatmap > 0.05
    out = frame.copy()
    out[mask] = cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)[mask]
    return out
