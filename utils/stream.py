"""
Stream capture utilities for ASU live webcam feeds.
Handles HLS/MJPEG stream ingestion with robust fallback.
"""

import cv2
import time
import threading
import numpy as np
from typing import Optional
import urllib.request
import ssl


# ASU Webcam stream URL
ASU_HAYDEN_URL = "https://view.asu.edu/tempe/hayden"

# We'll try multiple stream extraction strategies
STREAM_CANDIDATES = [
    # Direct HLS/RTSP attempts and common webcam embed patterns
    "https://view.asu.edu/tempe/hayden",
    "https://s3.amazonaws.com/asu-tempe-hayden/playlist.m3u8",  # placeholder
]


def try_extract_stream_url(page_url: str) -> Optional[str]:
    """
    Attempt to extract a direct stream URL from the webcam page.
    Returns None if extraction fails (dashboard will show static frame).
    """
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        req = urllib.request.Request(
            page_url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")

        # Look for common stream URL patterns
        import re
        patterns = [
            r'(https?://[^\s"\']+\.m3u8[^\s"\']*)',
            r'(rtsp://[^\s"\']+)',
            r'src=["\']([^"\']+\.mp4[^"\']*)["\']',
            r'file:\s*["\']([^"\']+)["\']',
        ]
        for pat in patterns:
            matches = re.findall(pat, html)
            if matches:
                return matches[0]
    except Exception as e:
        print(f"[Stream] Could not extract stream URL: {e}")
    return None


class StreamCapture:
    """
    Thread-safe video stream capture with automatic reconnection.
    Falls back to demo mode (local MP4) if stream is unavailable.
    """

    def __init__(
        self,
        stream_url: str,
        fps_target: int = 5,
        fallback_video: str = None,
        demo_mode: bool = False,
    ):
        self.stream_url   = stream_url
        self.fps_target   = fps_target
        self.fallback_video = fallback_video
        self.demo_mode    = demo_mode

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame: Optional[np.ndarray]     = None
        self._lock = threading.Lock()
        self._running = False
        self._thread  = None
        self._connected = False
        self._frame_count = 0
        self._last_error  = ""

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        # Wait briefly for first frame
        time.sleep(2)

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()

    def _open_capture(self) -> bool:
        """Try to open the stream; fall back to demo video if needed."""
        sources_to_try = []

        if not self.demo_mode:
            # Try live stream URL directly (works if ASU exposes an HLS endpoint)
            sources_to_try.append(self.stream_url)
            # Try extracting embedded stream URL
            extracted = try_extract_stream_url(self.stream_url)
            if extracted:
                sources_to_try.insert(0, extracted)

        # Always add fallback
        if self.fallback_video:
            sources_to_try.append(self.fallback_video)

        for src in sources_to_try:
            print(f"[Stream] Trying: {src[:80]}")
            cap = cv2.VideoCapture(src)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    self._cap = cap
                    self._connected = True
                    print(f"[Stream] Connected to: {src[:80]}")
                    return True
                cap.release()

        self._connected = False
        self._last_error = "Could not connect to any stream. Using demo frames."
        print(f"[Stream] {self._last_error}")
        return False

    def _capture_loop(self):
        interval = 1.0 / max(self.fps_target, 1)
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                if not self._open_capture():
                    time.sleep(5)
                    continue

            ret, frame = self._cap.read()
            if not ret:
                print("[Stream] Frame read failed. Reconnecting...")
                self._cap.release()
                self._cap = None
                self._connected = False
                time.sleep(2)
                continue

            with self._lock:
                self._frame = frame
                self._frame_count += 1

            time.sleep(interval)

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def last_error(self) -> str:
        return self._last_error
