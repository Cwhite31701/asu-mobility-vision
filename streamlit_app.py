"""
streamlit_app.py — ASU Mobility Vision (Streamlit Cloud entry point)

Cloud-safe version:
  • Uses pretrained YOLOv8n (auto-downloaded by ultralytics)
  • Pulls live ASU Hayden webcam or falls back to graceful demo
  • No local video files required
"""

import os
import sys
import time
import tempfile
from pathlib import Path
from collections import deque

# Tell FFmpeg (cv2.VideoCapture's backend) to impersonate a browser when
# fetching HLS from the ASU/Kinesis endpoints. AWS Kinesis Video's HLS host
# occasionally rejects plain Python clients; setting Referer + User-Agent
# to match the ASU page makes our requests look like the view.asu.edu SPA.
# MUST be set BEFORE any cv2.VideoCapture() call, which is why it's up here.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "referer;https://view.asu.edu/"
    "|user_agent;Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
)

import cv2
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import requests
import urllib.request
import ssl
import re

# ── Ensure utils is importable ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils.congestion import CongestionTracker
from utils.overlay import draw_detections, draw_hud, build_heatmap

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME        = "yolov8n.pt"          # downloaded automatically if absent
ASU_STREAM_URL    = "https://view.asu.edu/tempe/hayden"
# Direct HLS Media Playlist for the ASU Hayden cam (AWS Kinesis Video Streams).
# NOTE: The SessionToken in this URL is time-limited (typically ~5–60 min).
# When the live feed stops working, grab a fresh URL from DevTools → Network
# (filter: m3u8) on https://view.asu.edu/tempe/hayden and replace the string below.
ASU_HLS_URL = (
    "https://b-117b36f5.kinesisvideo.us-west-2.amazonaws.com/hls/v1/"
    "getHLSMediaPlaylist.m3u8"
    "?SessionToken=CiBYIXWN98f32H0XsJIoPGt_jm2e_EtuFhHbWWb2an7d7hIQ539FFZBT9FQCmp65ZGyAJxoZW3Inqv4kSM4ImX2kS1Xg3esnSYWPsbfOQSIgLW6yBJG0OMrLlZkYxyQk3NAjdWdFGhjGntY86UmFnr0~"
    "&TrackNumber=1"
)
HEATMAP_HISTORY   = 40
LOG_DIR           = Path("logs")
STATUS_EMOJI      = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}

# Demo image URLs (ASU campus stock — used as cloud fallback)
DEMO_IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/ASU_Hayden_Library.jpg/1280px-ASU_Hayden_Library.jpg",
]

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASU Mobility Vision",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: linear-gradient(135deg, #0a0e1a 0%, #0f1628 50%, #0d1420 100%); }
.metric-card {
    background: linear-gradient(135deg,rgba(255,255,255,.06),rgba(255,255,255,.02));
    border: 1px solid rgba(255,255,255,.1); border-radius:16px;
    padding:20px; text-align:center; backdrop-filter:blur(10px);
}
.metric-number { font-size:3rem; font-weight:700; line-height:1; }
.metric-label  { font-size:.8rem; color:#8892a4; text-transform:uppercase; letter-spacing:1px; margin-top:6px; }
.status-badge  { display:inline-block; padding:8px 20px; border-radius:50px;
    font-weight:700; font-size:1rem; letter-spacing:1px; text-transform:uppercase;
    width:100%; text-align:center; }
.status-low    { background:rgba(64,220,100,.15);  color:#40dc64; border:1px solid rgba(64,220,100,.4); }
.status-medium { background:rgba(255,200,30,.15);  color:#ffc81e; border:1px solid rgba(255,200,30,.4); }
.status-high   { background:rgba(255,59,59,.15);   color:#ff3b3b; border:1px solid rgba(255,59,59,.4); }
.section-title { font-size:.75rem; font-weight:600; color:#5b6a82;
    text-transform:uppercase; letter-spacing:2px; margin-bottom:12px; }
.score-ring    { font-size:2.2rem; font-weight:700; color:#a78bfa; }
.info-box { background:rgba(167,139,250,.07); border:1px dashed rgba(167,139,250,.3);
    border-radius:12px; padding:16px; font-size:.85rem; color:#8892a4; margin-bottom:12px;}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "running": False, "model": None, "tracker": None,
        "congestion": None, "history_df": pd.DataFrame(columns=["time","score","walkers","wheeled"]),
        "det_history": deque(maxlen=HEATMAP_HISTORY),
        "frame_idx": 0, "start_time": None,
        "show_heatmap": False, "show_track_ids": True,
        "conf_threshold": 0.40, "cap": None,
        # NEW: source routing
        "source_mode": "ASU Live (best-effort)",
        "custom_url": "",
        "uploaded_video_path": None,
        "use_tracker": True,
        "target_fps": 1,
        "window_sec": 60,
        # Performance knobs (see "Performance" section in sidebar)
        "speed_mode": "Balanced",
        "tick_count": 0,
        "last_dets": None,
        "last_counts": (0, 0),
        "last_metrics": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


@st.cache_resource(show_spinner="Downloading YOLOv8n… (first run only)")
def load_model():
    from ultralytics import YOLO
    return YOLO(MODEL_NAME)


def _fetch_url_bytes(url: str, timeout: int = 8) -> bytes | None:
    """HTTP GET with ASU-friendly headers; returns bytes or None."""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _decode_image_bytes(data: bytes) -> np.ndarray | None:
    try:
        arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _is_stream_url(url: str) -> bool:
    """True if the URL looks like a video stream (HLS/DASH/MP4/TS/RTSP)."""
    if not url:
        return False
    path = url.lower().split("?", 1)[0]
    if url.lower().startswith(("rtsp://", "rtmp://")):
        return True
    return any(path.endswith(ext) for ext in (".m3u8", ".mpd", ".mp4", ".ts", ".m4s"))


def _get_or_open_stream_cap(url: str) -> cv2.VideoCapture | None:
    """
    Cache a VideoCapture per URL in session state so we don't reconnect
    (and re-download the manifest + init segment) on every fragment tick.
    """
    key = f"_stream_cap::{url}"
    cap = st.session_state.get(key)
    if cap is not None and cap.isOpened():
        return cap
    try:
        new_cap = cv2.VideoCapture(url)
        if not new_cap.isOpened():
            new_cap.release()
            return None
        st.session_state[key] = new_cap
        return new_cap
    except Exception:
        return None


def fetch_frame_from_url(url: str) -> np.ndarray | None:
    """
    Fetch one frame from a URL. Handles:
      - direct JPEG/PNG/WEBP endpoints (one HTTP GET, decoded with cv2.imdecode)
      - HLS (.m3u8), DASH (.mpd), MP4, TS, RTSP (cached VideoCapture, one read per tick)
    """
    if not url:
        return None
    if _is_stream_url(url):
        cap = _get_or_open_stream_cap(url)
        if cap is None:
            return None
        ret, frame = cap.read()
        if ret:
            return frame
        # Stream may have stalled/ended — drop the cached cap so next tick reopens.
        key = f"_stream_cap::{url}"
        try:
            cap.release()
        except Exception:
            pass
        st.session_state.pop(key, None)
        return None
    # Otherwise treat as a still-image URL
    data = _fetch_url_bytes(url)
    if data is None:
        return None
    return _decode_image_bytes(data)


def record_hls_clip(url: str, out_path: str, duration_sec: int = 120) -> tuple[bool, str, int]:
    """
    Record ~duration_sec of live video from an HLS / MP4 / stream URL to an MP4 on disk.
    Designed for the ASU Kinesis HLS feed (tokens expire ~5–10 min, so keep clips short).

    Returns (ok, message, frames_written).

    UX: shows a Streamlit progress bar. Streamlit's event loop is blocked for the
    duration — don't touch other controls while recording.
    """
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return False, "Could not open stream — token may have expired. Grab a fresh .m3u8 URL.", 0

    # Probe first frame so we know geometry + that the stream is actually live
    ok_first, first = cap.read()
    if not ok_first or first is None:
        cap.release()
        return False, "Stream opened but no frames came through. Token likely expired.", 0

    h, w = first.shape[:2]
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = src_fps if (src_fps and 1 <= src_fps <= 60) else 5.0  # ASU cam is typically ~5 fps

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        cap.release()
        return False, f"Could not open writer for {out_path}", 0

    writer.write(first)  # don't waste the frame we probed

    progress  = st.progress(0.0, text=f"Recording 0 / {duration_sec}s …")
    deadline  = time.time() + duration_sec
    start_t   = time.time()
    frames    = 1
    stall_ct  = 0  # consecutive read failures

    while time.time() < deadline:
        ret, frame = cap.read()
        if not ret or frame is None:
            stall_ct += 1
            if stall_ct > 50:            # ~stream dead / token expired mid-record
                break
            time.sleep(0.1)
            continue
        stall_ct = 0
        writer.write(frame)
        frames += 1
        elapsed = time.time() - start_t
        pct = min(1.0, elapsed / duration_sec)
        progress.progress(
            pct,
            text=f"Recording {int(elapsed)} / {duration_sec}s  ·  {frames} frames @ {frames/max(elapsed,0.1):.1f} fps",
        )

    cap.release()
    writer.release()
    progress.empty()

    if frames < int(fps * 2):            # fewer than ~2s of video actually captured
        return False, f"Only captured {frames} frames before the stream died. Try a fresher token.", frames
    return True, f"Saved {frames} frames ({frames/fps:.0f}s @ {fps:.0f} fps) to {out_path}", frames


def try_live_stream(page_url: str) -> np.ndarray | None:
    """
    Best-effort ASU live-cam fetch. The ASU Views site is a JS SPA, so the
    image URL is usually injected client-side and won't appear in static HTML.
    We still try a handful of regex patterns — if the page ever serves a
    direct <img> or stream URL, we'll grab a frame from it.
    """
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(page_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=8) as r:
            html = r.read().decode("utf-8", errors="ignore")
        patterns = [
            # Only accept real stream/webcam sources, not Drupal marketing images.
            r'(https?://[^\s"\']+\.m3u8[^\s"\']*)',
            r'(https?://[^\s"\']+\.mpd[^\s"\']*)',
            r'src=["\']([^"\']+\.mp4[^"\']*)["\']',
            r'(https?://[^\s"\']+/(?:live|stream|cam|webcam|hayden)/[^\s"\']+\.(?:jpg|jpeg|png|m3u8|ts|m4s))',
        ]
        for pat in patterns:
            for candidate in re.findall(pat, html, flags=re.IGNORECASE):
                if candidate.startswith("//"):
                    candidate = "https:" + candidate
                elif candidate.startswith("/"):
                    candidate = "https://view.asu.edu" + candidate
                frame = fetch_frame_from_url(candidate)
                if frame is not None:
                    return frame
    except Exception:
        pass
    return None


def get_demo_frame() -> np.ndarray | None:
    """Return a static demo frame for cloud demo mode."""
    try:
        resp = requests.get(DEMO_IMAGE_URLS[0], timeout=8)
        img = _decode_image_bytes(resp.content)
        if img is not None:
            return img
    except Exception:
        pass
    # Last resort: synthetic grey frame with ASCII-safe text (OpenCV can't draw em-dashes)
    frame = np.full((480, 854, 3), 30, dtype=np.uint8)
    cv2.putText(frame, "ASU Mobility Vision - Demo Mode",
                (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (160,160,200), 2)
    return frame


# ── Source dispatcher ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _open_video_capture(path: str) -> cv2.VideoCapture:
    return cv2.VideoCapture(path)


def get_source_frame() -> tuple[np.ndarray | None, str]:
    """
    Dispatch based on source_mode. Returns (frame, source_label).
    """
    mode = st.session_state.get("source_mode", "ASU Live (best-effort)")

    if mode == "Uploaded video":
        path = st.session_state.get("uploaded_video_path")
        if path and Path(path).exists():
            cap = _open_video_capture(path)
            if cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the video
                    ret, frame = cap.read()
                if ret:
                    return frame, "UPLOADED VIDEO"
        return get_demo_frame(), "DEMO (no upload)"

    if mode == "Custom URL (image or stream)":
        url = (st.session_state.get("custom_url") or "").strip()
        if url:
            frame = fetch_frame_from_url(url)
            if frame is not None:
                label = "CUSTOM STREAM" if _is_stream_url(url) else "CUSTOM IMAGE"
                return frame, label
        return get_demo_frame(), "DEMO (no URL)"

    # Default: ASU Live best-effort.
    # 1) Try the known AWS Kinesis HLS endpoint directly — this is the real feed.
    frame = fetch_frame_from_url(ASU_HLS_URL)
    if frame is not None:
        return frame, "LIVE ASU HAYDEN"
    # 2) Fall back to scraping the ASU page for any stream/image URL it exposes.
    frame = try_live_stream(ASU_STREAM_URL)
    if frame is not None:
        return frame, "LIVE ASU HAYDEN"
    # 3) Last resort: demo image so the dashboard still has something to show.
    return get_demo_frame(), "DEMO FRAME"


# Speed-mode presets: (YOLO input size, run-inference-every-N-ticks, chart-refresh-every-N-ticks)
# Smaller imgsz → faster; bigger N → fewer expensive calls per second.
SPEED_PRESETS = {
    "Best Quality": {"imgsz": 640, "infer_every": 1, "chart_every": 3},
    "Balanced":     {"imgsz": 416, "infer_every": 2, "chart_every": 5},
    "Fastest":      {"imgsz": 320, "infer_every": 3, "chart_every": 8},
}


def run_yolo_inference(frame_bgr, model, conf, use_tracker, imgsz):
    """Run YOLO + (optional) SORT tracker. Returns (walkers, wheeled, display_dets)."""
    results = model(frame_bgr, conf=conf, verbose=False, imgsz=imgsz)[0]
    raw_dets = []
    for box in results.boxes:
        cls_id = int(box.cls[0].item())
        if cls_id not in (0, 1):
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        raw_dets.append({"x1":x1,"y1":y1,"x2":x2,"y2":y2,
                         "cls_id":cls_id,"conf":float(box.conf[0].item())})
    if use_tracker and st.session_state.tracker:
        display_dets = st.session_state.tracker.update(raw_dets)
    else:
        display_dets = raw_dets
    walkers = sum(1 for d in display_dets if d.get("cls_id", 0) == 0)
    wheeled = sum(1 for d in display_dets if d.get("cls_id", 0) == 1)
    return walkers, wheeled, display_dets


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown('<div class="info-box">🌐 <strong>Cloud Demo</strong><br>Pick a video source below, then press <strong>Start</strong>. <strong>ASU Live</strong> uses the baked-in HLS feed; if the AWS session token expires, paste a fresh <code>.m3u8</code> URL into <strong>Custom URL</strong>.</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-title">Video Source</div>', unsafe_allow_html=True)
    SOURCE_OPTIONS = ["ASU Live (best-effort)", "Custom URL (image or stream)", "Uploaded video"]
    st.session_state.source_mode = st.radio(
        "Source",
        SOURCE_OPTIONS,
        index=SOURCE_OPTIONS.index(st.session_state.get("source_mode", SOURCE_OPTIONS[0])),
        label_visibility="collapsed",
    )

    if st.session_state.source_mode == "Custom URL (image or stream)":
        st.session_state.custom_url = st.text_input(
            "URL",
            value=st.session_state.get("custom_url", ""),
            placeholder="https://…/hayden.m3u8   or   https://…/latest.jpg",
            help=("Accepts direct images (.jpg/.png/.webp) or live streams "
                  "(.m3u8 HLS, .mpd DASH, .mp4, .ts, rtsp://). "
                  "Find the URL in DevTools → Network → filter by 'Media' or '.m3u8'."),
        )
    elif st.session_state.source_mode == "Uploaded video":
        uploaded = st.file_uploader("Upload mp4 / mov / avi", type=["mp4", "mov", "avi", "mkv"])
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.flush()
            tmp.close()
            st.session_state.uploaded_video_path = tmp.name
            st.success(f"Loaded: {uploaded.name}")

    # ── Record-from-live helper ──────────────────────────────────────────────
    # Captures N seconds of the live HLS stream to a local MP4 so the demo
    # keeps working after the AWS SessionToken expires. Great for demo day:
    # record right before you present, then flip to "Uploaded video" playback.
    st.markdown('<div class="section-title">Record Clip From Live</div>', unsafe_allow_html=True)
    rec_c1, rec_c2 = st.columns([2, 1])
    with rec_c1:
        rec_secs = st.number_input("Duration (sec)", min_value=10, max_value=300, value=120, step=10,
                                   help="Keep it under the token lifetime (~5 min).")
    with rec_c2:
        st.markdown("<br>", unsafe_allow_html=True)
        record_clicked = st.button("⏺ Record", use_container_width=True)

    if record_clicked:
        # Prefer the custom URL if the user pasted a fresh one; else use the baked HLS URL.
        rec_url = (st.session_state.get("custom_url") or "").strip() or ASU_HLS_URL
        if not rec_url:
            st.error("No stream URL available. Paste a fresh .m3u8 into Custom URL first.")
        else:
            out_path = str(Path(tempfile.gettempdir()) / f"asu_cap_{int(time.time())}.mp4")
            st.info(f"Recording {rec_secs}s — UI is frozen until it finishes.")
            ok, msg, nframes = record_hls_clip(rec_url, out_path, duration_sec=int(rec_secs))
            if ok:
                st.success(msg)
                # Drop any cached VideoCapture for the live URL so later plays don't collide
                st.session_state.pop(f"_stream_cap::{rec_url}", None)
                # Auto-switch to playback mode so pressing Start uses the clip
                st.session_state.uploaded_video_path = out_path
                st.session_state.source_mode = "Uploaded video"
                st.rerun()
            else:
                st.error(msg)

    st.markdown("---")
    st.markdown('<div class="section-title">Detection Settings</div>', unsafe_allow_html=True)
    st.session_state.conf_threshold = st.slider("Confidence Threshold", 0.20, 0.80, 0.40, 0.05)
    st.session_state.use_tracker    = st.toggle("SORT Object Tracking", value=st.session_state.get("use_tracker", True))
    st.session_state.show_heatmap   = st.toggle("Density Heatmap",  value=st.session_state.get("show_heatmap", False))
    st.session_state.show_track_ids = st.toggle("Show Track IDs",   value=st.session_state.get("show_track_ids", True))
    st.session_state.target_fps     = st.slider("Target FPS (hint)", 1, 4, st.session_state.get("target_fps", 1),
                                                help="Streamlit Cloud CPU caps practical rate around 1–2 FPS for YOLOv8n.")

    st.markdown('<div class="section-title">Performance</div>', unsafe_allow_html=True)
    st.session_state.speed_mode = st.radio(
        "Speed mode",
        list(SPEED_PRESETS.keys()),
        index=list(SPEED_PRESETS.keys()).index(st.session_state.get("speed_mode", "Balanced")),
        horizontal=True,
        label_visibility="collapsed",
        help=(
            "Best Quality: 640px YOLO input, detection every tick. Sharpest boxes, slower.\n"
            "Balanced: 416px, detection every 2 ticks. Smoother playback (recommended).\n"
            "Fastest: 320px, detection every 3 ticks. Snappy video, looser boxes."
        ),
    )

    st.markdown('<div class="section-title">Congestion Window</div>', unsafe_allow_html=True)
    st.session_state.window_sec = st.slider("Rolling Window (sec)", 10, 120, st.session_state.get("window_sec", 60), 10)

    st.markdown("---")
    col_run, col_stop = st.columns(2)
    with col_run:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            try:
                st.session_state.model = load_model()
            except Exception as e:
                st.error(f"Model load failed: {e}")
                st.stop()
            if st.session_state.use_tracker:
                from utils.tracker import SORTTracker, KalmanBoxTracker
                KalmanBoxTracker.count = 0
                st.session_state.tracker = SORTTracker()
            else:
                st.session_state.tracker = None
            LOG_DIR.mkdir(exist_ok=True)
            st.session_state.congestion  = CongestionTracker(window_seconds=st.session_state.window_sec, log_dir=str(LOG_DIR))
            st.session_state.running     = True
            st.session_state.start_time  = time.time()
            st.session_state.frame_idx   = 0
            st.session_state.history_df  = pd.DataFrame(columns=["time","score","walkers","wheeled"])
            st.session_state.det_history.clear()
            # Reset perf-loop state so inference fires on the very first tick
            st.session_state.tick_count   = 0
            st.session_state.last_dets    = None
            st.session_state.last_counts  = (0, 0)
            st.session_state.last_metrics = None
    with col_stop:
        if st.button("⏹ Stop", use_container_width=True):
            st.session_state.running = False

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <strong>Congestion Formula</strong><br>
    Score = Walkers × 1.0 + Wheeled × 1.5<br><br>
    🟢 <strong>Low</strong> &lt; 5 &nbsp;|&nbsp;
    🟡 <strong>Medium</strong> 5–12 &nbsp;|&nbsp;
    🔴 <strong>High</strong> &gt; 12
    </div>
    """, unsafe_allow_html=True)

    # Log download
    log_files = sorted(LOG_DIR.glob("*.csv")) if LOG_DIR.exists() else []
    if log_files:
        with open(log_files[-1], "rb") as f:
            st.download_button("⬇ Download Session CSV", f,
                               file_name=log_files[-1].name,
                               mime="text/csv", use_container_width=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏫 ASU Mobility Vision")
st.markdown("*Real-time pedestrian & wheeled mobility detection · Hayden Library Zone · CIS 515*")

main_col, side_col = st.columns([3, 1], gap="large")

with main_col:
    video_ph = st.empty()

with side_col:
    st.markdown('<div class="section-title">Live Counts</div>', unsafe_allow_html=True)
    walker_ph  = st.empty()
    wheeled_ph = st.empty()
    score_ph   = st.empty()
    status_ph  = st.empty()
    st.markdown("---")
    st.markdown('<div class="section-title">Rolling Avg & Forecast</div>', unsafe_allow_html=True)
    rolling_ph = st.empty()
    predict_ph = st.empty()
    st.markdown("---")
    session_ph = st.empty()

st.markdown("---")
ch1, ch2 = st.columns([2, 1])
with ch1:
    st.markdown('<div class="section-title">Congestion Over Time</div>', unsafe_allow_html=True)
    chart_ph = st.empty()
with ch2:
    st.markdown('<div class="section-title">Walker vs Wheeled Split</div>', unsafe_allow_html=True)
    pie_ph = st.empty()


# ── Render helpers ────────────────────────────────────────────────────────────
def r_metric(c, val, label, color):
    c.markdown(f'<div class="metric-card"><div class="metric-number" style="color:{color}">{val}</div>'
               f'<div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

def r_status(c, status):
    emoji = STATUS_EMOJI.get(status, "⚪")
    c.markdown(f'<div class="status-badge status-{status.lower()}">{emoji} {status}</div>',
               unsafe_allow_html=True)

def r_chart(c, df):
    if df.empty or len(df) < 2:
        c.info("Collecting data…"); return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["time"], y=df["score"], mode="lines", name="Score",
        line=dict(color="#a78bfa", width=2.5), fill="tozeroy", fillcolor="rgba(167,139,250,.1)"))
    fig.add_trace(go.Scatter(x=df["time"], y=df["walkers"], mode="lines", name="Walkers",
        line=dict(color="#40dc9f", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=df["time"], y=df["wheeled"], mode="lines", name="Wheeled",
        line=dict(color="#ffb830", width=1.5, dash="dot")))
    fig.add_hline(y=5,  line_dash="dash", line_color="rgba(64,220,100,.4)",  annotation_text="Low")
    fig.add_hline(y=12, line_dash="dash", line_color="rgba(255,59,59,.4)",   annotation_text="High")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8892a4", size=11), margin=dict(l=20,r=20,t=20,b=20),
        legend=dict(orientation="h", y=-0.2), height=240,
        xaxis=dict(showgrid=False, title="Elapsed (s)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.05)"))
    c.plotly_chart(fig, use_container_width=True)

def r_pie(c, walkers, wheeled):
    if walkers + wheeled == 0:
        c.info("No detections yet."); return
    fig = go.Figure(go.Pie(labels=["Walkers","Wheeled"], values=[walkers, wheeled], hole=0.55,
        marker=dict(colors=["#40dc9f","#ffb830"]), textinfo="label+percent"))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#8892a4",size=11),
        margin=dict(l=10,r=10,t=10,b=10), showlegend=False, height=240)
    c.plotly_chart(fig, use_container_width=True)


# ── Main loop (runs as a Streamlit fragment — panel-scoped refresh, no flicker) ─
#
# Cadence strategy (for smooth playback on Streamlit Cloud's 1-vCPU free tier):
#   • Fragment fires every 0.5s  → video feels 2x smoother than before.
#   • YOLO only runs every Nth tick (per speed preset); other ticks reuse the
#     last detections and just redraw them on the fresh frame. Boxes drift
#     slightly between inference ticks but the video moves cleanly.
#   • Plotly charts (expensive to rebuild) refresh every Mth tick only.
@st.fragment(run_every="0.5s")
def video_step():
    if not st.session_state.running:
        return

    model = st.session_state.model
    cong  = st.session_state.congestion
    if model is None or cong is None:
        return

    preset       = SPEED_PRESETS[st.session_state.get("speed_mode", "Balanced")]
    imgsz        = preset["imgsz"]
    infer_every  = preset["infer_every"]
    chart_every  = preset["chart_every"]

    tick = int(st.session_state.get("tick_count", 0))
    st.session_state.tick_count = tick + 1

    frame, src_label = get_source_frame()
    if frame is None:
        video_ph.warning("⚠️ No frame available — check your source selection.")
        return

    # Only run YOLO on an "inference tick" (or on the very first tick so we
    # have something to draw). Otherwise reuse cached detections.
    do_infer = (tick % infer_every == 0) or (st.session_state.get("last_dets") is None)

    if do_infer:
        walkers, wheeled, dets = run_yolo_inference(
            frame, model,
            st.session_state.conf_threshold,
            st.session_state.use_tracker,
            imgsz,
        )
        metrics = cong.update(walkers, wheeled)
        st.session_state.last_dets    = dets
        st.session_state.last_counts  = (walkers, wheeled)
        st.session_state.last_metrics = metrics
        st.session_state.det_history.append(dets)
    else:
        dets      = st.session_state.get("last_dets") or []
        walkers, wheeled = st.session_state.get("last_counts", (0, 0))
        metrics   = st.session_state.get("last_metrics")
        if metrics is None:                      # shouldn't happen, but guard
            return

    # Redraw boxes on the fresh frame (cheap) so the video always looks live.
    annotated = draw_detections(
        frame, dets, show_conf=True, show_track_ids=st.session_state.show_track_ids,
    )

    if st.session_state.show_heatmap and len(st.session_state.det_history) > 5:
        annotated = build_heatmap(annotated, list(st.session_state.det_history))

    annotated = draw_hud(
        annotated, walkers=walkers, wheeled=wheeled,
        score=metrics["score"], status=metrics["status"],
        source_label=src_label, rolling_avg=metrics["rolling_avg_score"],
    )

    video_ph.image(
        cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        channels="RGB", use_container_width=True,
    )

    # Cheap DOM updates — run every tick.
    r_metric(walker_ph,  walkers, "Walkers 🚶",  "#40dc9f")
    r_metric(wheeled_ph, wheeled, "Wheeled 🚲",  "#ffb830")
    score_ph.markdown(
        f'<div class="metric-card"><div class="score-ring">{metrics["score"]}</div>'
        f'<div class="metric-label">Congestion Score</div></div>', unsafe_allow_html=True)
    r_status(status_ph, metrics["status"])
    rolling_ph.markdown(
        f'<div class="metric-card"><div style="font-size:1.8rem;font-weight:700;color:#c4b5fd">'
        f'{metrics["rolling_avg_score"]}</div>'
        f'<div class="metric-label">Rolling Avg ({st.session_state.window_sec}s)</div></div>',
        unsafe_allow_html=True)

    elapsed = int(time.time() - (st.session_state.start_time or time.time()))
    session_ph.markdown(
        f'<div style="font-size:.82rem;color:#5b6a82;line-height:1.8">'
        f'⏱ {elapsed}s &nbsp;|&nbsp; 🎞 Frame #{st.session_state.frame_idx}<br>'
        f'📊 {metrics["window_size"]} samples in window<br>'
        f'🔗 Source: {src_label} &nbsp;|&nbsp; ⚙️ {st.session_state.speed_mode}</div>',
        unsafe_allow_html=True)

    # Only append a history row on real inference ticks — otherwise we'd
    # duplicate the same (walkers, wheeled) every 500ms and pollute the chart.
    if do_infer:
        new_row = pd.DataFrame([{
            "time": metrics["elapsed_seconds"],
            "score": metrics["score"],
            "walkers": walkers, "wheeled": wheeled,
        }])
        st.session_state.history_df = pd.concat(
            [st.session_state.history_df, new_row], ignore_index=True,
        ).tail(500)

    st.session_state.frame_idx += 1

    # Expensive Plotly rebuilds — throttle to every chart_every ticks.
    if tick % chart_every == 0:
        pred = cong.predict_congestion(300)
        if pred["predicted_score"] is not None:
            icon = {"increasing":"📈","decreasing":"📉","stable":"➡️"}.get(pred["trend"],"➡️")
            predict_ph.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size:.75rem;color:#5b6a82;text-transform:uppercase;letter-spacing:1px">5-min Forecast</div>'
                f'<div style="font-size:1.5rem;font-weight:700;color:#f9a8d4">{pred["predicted_score"]}</div>'
                f'<div style="font-size:.8rem;color:#8892a4">{icon} {pred["trend"].capitalize()} · {pred["predicted_status"]}</div>'
                f'</div>', unsafe_allow_html=True)
        r_chart(chart_ph, st.session_state.history_df)
        r_pie(pie_ph, walkers, wheeled)


if st.session_state.running:
    video_step()
else:
    video_ph.markdown("""
    <div style="background:linear-gradient(135deg,rgba(167,139,250,.08),rgba(96,165,250,.05));
        border:1px dashed rgba(167,139,250,.3); border-radius:16px; padding:60px;
        text-align:center; color:#5b6a82;">
        <div style="font-size:4rem;margin-bottom:16px">🏫</div>
        <div style="font-size:1.3rem;font-weight:600;color:#8892a4;margin-bottom:8px">
            ASU Mobility Vision — CIS 515
        </div>
        <div style="font-size:.95rem">
            Press <strong style="color:#a78bfa">▶ Start</strong> in the sidebar to begin real-time detection.
        </div>
    </div>""", unsafe_allow_html=True)
    for ph in [walker_ph, wheeled_ph]:
        ph.markdown('<div class="metric-card"><div class="metric-number" style="color:#5b6a82">—</div>'
                    '<div class="metric-label">Waiting</div></div>', unsafe_allow_html=True)
    score_ph.markdown('<div class="metric-card"><div class="score-ring">—</div>'
                      '<div class="metric-label">Congestion Score</div></div>', unsafe_allow_html=True)
    status_ph.markdown('<div class="status-badge" style="background:rgba(255,255,255,.05);'
                       'color:#5b6a82;border:1px solid rgba(255,255,255,.08)">⚪ IDLE</div>',
                       unsafe_allow_html=True)
