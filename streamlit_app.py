"""
streamlit_app.py — ASU Mobility Vision (Streamlit Cloud entry point)

Cloud-safe version:
  • Uses pretrained YOLOv8n (auto-downloaded by ultralytics)
  • Pulls live ASU Hayden webcam or falls back to graceful demo
  • No local video files required
"""

import sys
import time
from pathlib import Path
from collections import deque

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
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


@st.cache_resource(show_spinner="Downloading YOLOv8n… (first run only)")
def load_model():
    from ultralytics import YOLO
    return YOLO(MODEL_NAME)


def try_live_stream(url: str):
    """Attempt to grab one frame from ASU live stream."""
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=8) as r:
            html = r.read().decode("utf-8", errors="ignore")
        for pat in [r'(https?://[^\s"\']+\.m3u8[^\s"\']*)', r'src=["\']([^"\']+\.mp4[^"\']*)["\']']:
            m = re.findall(pat, html)
            if m:
                cap = cv2.VideoCapture(m[0])
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        return frame
    except Exception:
        pass
    return None


def get_demo_frame() -> np.ndarray | None:
    """Return a synthetic demo frame for cloud demo mode."""
    # Try to fetch a static demo image via requests
    try:
        resp = requests.get(DEMO_IMAGE_URLS[0], timeout=8)
        arr  = np.frombuffer(resp.content, np.uint8)
        img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    # Last resort: generate a synthetic grey frame with text
    frame = np.full((480, 854, 3), 30, dtype=np.uint8)
    cv2.putText(frame, "ASU Mobility Vision — Demo Mode",
                (60, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (160,160,200), 2)
    return frame


def process_frame(frame_bgr, model, conf, use_tracker):
    results = model(frame_bgr, conf=conf, verbose=False)[0]
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

    walkers = sum(1 for d in display_dets if d.get("cls_id",0)==0)
    wheeled = sum(1 for d in display_dets if d.get("cls_id",0)==1)
    annotated = draw_detections(frame_bgr, display_dets,
                                show_conf=True,
                                show_track_ids=st.session_state.show_track_ids)
    return annotated, walkers, wheeled, display_dets


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    st.markdown('<div class="info-box">🌐 <strong>Cloud Demo</strong><br>Using pretrained YOLOv8n + ASU live stream (falls back to demo frame if stream is unavailable).</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-title">Detection Settings</div>', unsafe_allow_html=True)
    st.session_state.conf_threshold = st.slider("Confidence Threshold", 0.20, 0.80, 0.40, 0.05)
    use_tracker  = st.toggle("SORT Object Tracking", value=True)
    st.session_state.show_heatmap   = st.toggle("Density Heatmap",  value=False)
    st.session_state.show_track_ids = st.toggle("Show Track IDs",   value=True)
    target_fps   = st.slider("Target FPS", 1, 10, 3)

    st.markdown('<div class="section-title">Congestion Window</div>', unsafe_allow_html=True)
    window_sec = st.slider("Rolling Window (sec)", 10, 120, 60, 10)

    st.markdown("---")
    col_run, col_stop = st.columns(2)
    with col_run:
        if st.button("▶ Start", use_container_width=True, type="primary"):
            try:
                st.session_state.model = load_model()
            except Exception as e:
                st.error(f"Model load failed: {e}")
                st.stop()
            if use_tracker:
                from utils.tracker import SORTTracker, KalmanBoxTracker
                KalmanBoxTracker.count = 0
                st.session_state.tracker = SORTTracker()
            else:
                st.session_state.tracker = None
            LOG_DIR.mkdir(exist_ok=True)
            st.session_state.congestion  = CongestionTracker(window_seconds=window_sec, log_dir=str(LOG_DIR))
            st.session_state.running     = True
            st.session_state.start_time  = time.time()
            st.session_state.frame_idx   = 0
            st.session_state.history_df  = pd.DataFrame(columns=["time","score","walkers","wheeled"])
            st.session_state.det_history.clear()
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


# ── Main loop ─────────────────────────────────────────────────────────────────
if st.session_state.running:
    model = st.session_state.model
    cong  = st.session_state.congestion

    # Grab frame — try live stream first, fall back to demo
    frame = try_live_stream(ASU_STREAM_URL)
    src_label = "LIVE · ASU HAYDEN" if frame is not None else "DEMO FRAME"
    if frame is None:
        frame = get_demo_frame()

    if frame is not None and model is not None:
        annotated, walkers, wheeled, dets = process_frame(
            frame, model, st.session_state.conf_threshold, use_tracker)

        st.session_state.det_history.append(dets)

        if st.session_state.show_heatmap and len(st.session_state.det_history) > 5:
            annotated = build_heatmap(annotated, list(st.session_state.det_history))

        metrics  = cong.update(walkers, wheeled)
        annotated = draw_hud(annotated, walkers=walkers, wheeled=wheeled,
                             score=metrics["score"], status=metrics["status"],
                             source_label=src_label, rolling_avg=metrics["rolling_avg_score"])

        video_ph.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                       channels="RGB", use_container_width=True)

        r_metric(walker_ph,  walkers, "Walkers 🚶",  "#40dc9f")
        r_metric(wheeled_ph, wheeled, "Wheeled 🚲",  "#ffb830")
        score_ph.markdown(f'<div class="metric-card"><div class="score-ring">{metrics["score"]}</div>'
                          f'<div class="metric-label">Congestion Score</div></div>', unsafe_allow_html=True)
        r_status(status_ph, metrics["status"])

        rolling_ph.markdown(
            f'<div class="metric-card"><div style="font-size:1.8rem;font-weight:700;color:#c4b5fd">'
            f'{metrics["rolling_avg_score"]}</div>'
            f'<div class="metric-label">Rolling Avg ({window_sec}s)</div></div>',
            unsafe_allow_html=True)

        pred = cong.predict_congestion(300)
        if pred["predicted_score"] is not None:
            icon = {"increasing":"📈","decreasing":"📉","stable":"➡️"}.get(pred["trend"],"➡️")
            predict_ph.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size:.75rem;color:#5b6a82;text-transform:uppercase;letter-spacing:1px">5-min Forecast</div>'
                f'<div style="font-size:1.5rem;font-weight:700;color:#f9a8d4">{pred["predicted_score"]}</div>'
                f'<div style="font-size:.8rem;color:#8892a4">{icon} {pred["trend"].capitalize()} · {pred["predicted_status"]}</div>'
                f'</div>', unsafe_allow_html=True)

        elapsed = int(time.time() - (st.session_state.start_time or time.time()))
        session_ph.markdown(
            f'<div style="font-size:.82rem;color:#5b6a82;line-height:1.8">'
            f'⏱ {elapsed}s &nbsp;|&nbsp; 🎞 Frame #{st.session_state.frame_idx}<br>'
            f'📊 {metrics["window_size"]} samples in window<br>'
            f'🔗 Source: {src_label}</div>', unsafe_allow_html=True)

        new_row = pd.DataFrame([{"time": metrics["elapsed_seconds"],
                                  "score": metrics["score"],
                                  "walkers": walkers, "wheeled": wheeled}])
        st.session_state.history_df = pd.concat(
            [st.session_state.history_df, new_row], ignore_index=True).tail(500)
        st.session_state.frame_idx += 1

        r_chart(chart_ph, st.session_state.history_df)
        r_pie(pie_ph, walkers, wheeled)

        if metrics["status"] == "HIGH":
            st.toast(f"🔴 HIGH congestion! Score: {metrics['score']}", icon="🚨")

        time.sleep(1.0 / max(target_fps, 1))
        st.rerun()

    else:
        video_ph.warning("⚠️ Cannot load frame — check stream or demo fallback.")
        time.sleep(2); st.rerun()

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
