"""Worker manager for continuous, motion-gated CCTV monitoring.

Starts one background thread per sector. Each thread reads its video
continuously and, whenever MOG2 detects motion, runs a burst of YOLO + OCR
reads (see app/cctv.py). Plates found are written to the database.

Within a single motion burst the same plate may be read several times (from
different angles); we keep only the highest-confidence reading per plate in
that burst before writing, so the stored confidence reflects the best capture.

A separate timer thread periodically expires plates not seen for 24h.
"""

import os
import threading
import time
from datetime import datetime

from app import cctv, database

# How often to run the expiry cleanup (seconds).
EXPIRY_CHECK_SECONDS = int(os.environ.get("EXPIRY_CHECK_SECONDS", 600))

_threads: list[threading.Thread] = []
_stop = threading.Event()
_stats = {"reads": 0, "sectors": [], "started_at": None, "last_expiry": None}
_stats_lock = threading.Lock()


def _handle_plate(sector_id):
    """Build an on_plate callback that records a plate for this sector.

    cctv._capture_burst already dedups a burst to unique plates (best OCR
    score per plate) before calling this, so here we simply upsert each plate.
    """
    def on_plate(plate, confidence, ocr_score):
        database.record_detection(plate, sector_id, confidence, ocr_score)
        with _stats_lock:
            _stats["reads"] += 1

    return on_plate


def _expiry_loop():
    while not _stop.is_set():
        try:
            removed = database.expire_old()
            with _stats_lock:
                _stats["last_expiry"] = {
                    "at": datetime.now().isoformat(timespec="seconds"),
                    "removed": removed,
                }
        except Exception as exc:
            print(f"[scheduler] expiry error: {exc}")
        _stop.wait(EXPIRY_CHECK_SECONDS)


def start():
    """Start one monitor thread per sector plus the expiry timer (idempotent)."""
    if _threads:
        return

    _stop.clear()
    sectors = cctv.list_sectors()

    with _stats_lock:
        _stats["sectors"] = [s for s, _ in sectors]
        _stats["started_at"] = datetime.now().isoformat(timespec="seconds")

    for sector_id, video_path in sectors:
        t = threading.Thread(
            target=cctv.monitor_sector,
            args=(sector_id, video_path, _handle_plate(sector_id), _stop.is_set),
            daemon=True,
            name=f"monitor-{sector_id}",
        )
        t.start()
        _threads.append(t)

    expiry = threading.Thread(target=_expiry_loop, daemon=True, name="expiry")
    expiry.start()
    _threads.append(expiry)


def stop():
    """Signal all worker threads to stop."""
    _stop.set()


def stats() -> dict:
    """Return a snapshot of worker stats for the /status endpoint."""
    with _stats_lock:
        return dict(_stats)
