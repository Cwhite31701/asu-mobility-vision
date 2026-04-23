"""
Congestion metrics and rolling statistics for mobility analysis.
"""

import time
import csv
import os
from collections import deque
from pathlib import Path
import numpy as np


# Congestion scoring weights
WALKER_WEIGHT   = 1.0
WHEELED_WEIGHT  = 1.5

# Thresholds
LOW_THRESHOLD    = 5.0
MEDIUM_THRESHOLD = 12.0
# Above medium = HIGH


class CongestionTracker:
    """Tracks congestion metrics over a rolling window."""

    def __init__(self, window_seconds: int = 60, log_dir: str = None):
        self.window_seconds = window_seconds
        self._history: deque = deque()   # (timestamp, walkers, wheeled, score)
        self._start_time = time.time()

        # Logging
        self.log_dir = Path(log_dir) if log_dir else None
        self._csv_path = None
        self._csv_writer = None
        self._csv_file = None
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            self._csv_path = self.log_dir / f"mobility_log_{ts}.csv"
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(["timestamp", "walkers", "wheeled", "congestion_score", "status"])

    def __del__(self):
        if self._csv_file:
            try:
                self._csv_file.close()
            except Exception:
                pass

    @staticmethod
    def compute_score(walkers: int, wheeled: int) -> float:
        return walkers * WALKER_WEIGHT + wheeled * WHEELED_WEIGHT

    @staticmethod
    def get_status(score: float) -> str:
        if score < LOW_THRESHOLD:
            return "LOW"
        elif score < MEDIUM_THRESHOLD:
            return "MEDIUM"
        else:
            return "HIGH"

    def update(self, walkers: int, wheeled: int) -> dict:
        """Record a new observation and return current metrics."""
        now = time.time()
        score = self.compute_score(walkers, wheeled)
        status = self.get_status(score)

        self._history.append((now, walkers, wheeled, score))

        # Prune old entries outside the window
        cutoff = now - self.window_seconds
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        # Rolling averages
        scores = [e[3] for e in self._history]
        walker_counts = [e[1] for e in self._history]
        wheeled_counts = [e[2] for e in self._history]

        rolling_avg_score   = float(np.mean(scores)) if scores else 0.0
        rolling_avg_walkers = float(np.mean(walker_counts)) if walker_counts else 0.0
        rolling_avg_wheeled = float(np.mean(wheeled_counts)) if wheeled_counts else 0.0
        rolling_status      = self.get_status(rolling_avg_score)

        # Write to CSV log
        if self._csv_writer:
            self._csv_writer.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                walkers, wheeled, round(score, 2), status
            ])
            self._csv_file.flush()

        elapsed = now - self._start_time

        return {
            "timestamp": now,
            "elapsed_seconds": elapsed,
            "walkers": walkers,
            "wheeled": wheeled,
            "score": round(score, 2),
            "status": status,
            "rolling_avg_score": round(rolling_avg_score, 2),
            "rolling_avg_walkers": round(rolling_avg_walkers, 1),
            "rolling_avg_wheeled": round(rolling_avg_wheeled, 1),
            "rolling_status": rolling_status,
            "window_size": len(self._history),
        }

    def get_history_for_chart(self) -> dict:
        """Return lists of timestamps and scores for plotting."""
        if not self._history:
            return {"times": [], "scores": [], "walkers": [], "wheeled": []}

        t0 = self._history[0][0]
        times   = [round(e[0] - t0, 1) for e in self._history]
        scores  = [round(e[3], 2)       for e in self._history]
        walkers = [e[1]                  for e in self._history]
        wheeled = [e[2]                  for e in self._history]
        return {"times": times, "scores": scores, "walkers": walkers, "wheeled": wheeled}

    def predict_congestion(self, lookahead_seconds: int = 300) -> dict:
        """
        Simple linear trend extrapolation to predict congestion.
        Returns predicted score and status in `lookahead_seconds`.
        """
        if len(self._history) < 5:
            return {"predicted_score": None, "predicted_status": "UNKNOWN", "trend": "stable"}

        t0 = self._history[0][0]
        times  = np.array([e[0] - t0 for e in self._history])
        scores = np.array([e[3]       for e in self._history])

        # Linear regression
        coeffs = np.polyfit(times, scores, 1)
        slope  = coeffs[0]
        future_time = times[-1] + lookahead_seconds
        pred_score  = max(0.0, float(np.polyval(coeffs, future_time)))

        if slope > 0.05:
            trend = "increasing"
        elif slope < -0.05:
            trend = "decreasing"
        else:
            trend = "stable"

        return {
            "predicted_score": round(pred_score, 2),
            "predicted_status": self.get_status(pred_score),
            "trend": trend,
            "slope": round(float(slope), 4),
        }
