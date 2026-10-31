"""CCTV footage access + motion-gated plate capture.

Footage layout (configurable via the VIDEOS_DIR env var), one folder per
sector:

    videos/
      A1/  cam.mp4      <- sector A1 camera
      A2/  cam.mp4      <- sector A2 camera
      ...

Instead of sampling fixed frames, each sector is read continuously. A cheap
MOG2 background subtractor measures how much of the frame is moving. Only
when motion crosses a threshold do we run the expensive YOLO + OCR step,
and then we capture a short BURST of frames so a passing plate is read from
several angles. The best (highest-confidence) reading per plate wins.
"""

import os
from pathlib import Path

import cv2
import numpy as np

from app import detector

VIDEOS_DIR = Path(os.environ.get("VIDEOS_DIR", Path(__file__).parent.parent / "videos"))
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

# --- tunables (env-configurable) ---
# Process every Kth frame for motion (cheap). Lower = more responsive, heavier.
FRAME_SKIP = int(os.environ.get("FRAME_SKIP", 5))
# Fraction of the frame (0-1) that must be "moving" to count as motion.
MOTION_THRESHOLD = float(os.environ.get("MOTION_THRESHOLD", 0.02))
# When motion fires, read AT MOST this many frames in the burst.
BURST_FRAMES = int(os.environ.get("BURST_FRAMES", 12))
# A burst keeps reading while plates are still found; it stops early after
# this many consecutive frames with no plate (handles flickering detections).
BURST_MISS_LIMIT = int(os.environ.get("BURST_MISS_LIMIT", 5))
# After a burst, skip this many FRAMES before motion can trigger again.
# Frame-based (not wall-clock) so it behaves the same for files and streams.
# A wall-clock cooldown would swallow hundreds of video frames on fast CPUs.
MOTION_COOLDOWN_FRAMES = int(os.environ.get("MOTION_COOLDOWN_FRAMES", 15))
# YOLO confidence threshold.
DETECT_CONF = float(os.environ.get("DETECT_CONF", 0.25))


def list_sectors():
    """Return a list of (sector_id, video_path) for every sector folder."""
    sectors = []
    if not VIDEOS_DIR.exists():
        return sectors
    for folder in sorted(VIDEOS_DIR.iterdir()):
        if not folder.is_dir():
            continue
        videos = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in VIDEO_EXTS]
        if videos:
            sectors.append((folder.name, videos[0]))
    return sectors


def _new_subtractor():
    # detectShadows=False: we don't care about shadows, and it's faster.
    return cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=25, detectShadows=False
    )


def _motion_fraction(submask) -> float:
    """Fraction of foreground (moving) pixels in a MOG2 mask."""
    moving = int(np.count_nonzero(submask))
    total = submask.size
    return moving / total if total else 0.0


def _capture_burst(cap, on_plate):
    """Run YOLO over BURST_FRAMES consecutive frames, then emit unique plates.

    Across the burst the SAME plate is usually read several times with
    different scores (motion blur, angle changes). We dedup by plate and keep
    only the HIGHEST score per plate, then emit every unique plate once. We do
    NOT cap the count: if five different cars pass in one burst, all five
    unique plates are emitted -- the dedup only removes repeat reads of the
    same car, never different cars.

    Returns the number of frames actually processed.
    """
    processed = 0
    misses = 0  # consecutive frames with no plate found
    # plate -> (yolo_conf, ocr_score) of the best read so far. We rank by OCR
    # score because that reflects how trustworthy the TEXT is.
    best: dict[str, tuple] = {}

    # Keep reading up to BURST_FRAMES, but stop early only after several
    # consecutive frames with no plate. This "rides out" flickering: a plate
    # that appears in frames 1, 4, 7 is still captured even if 2, 3, 5, 6 miss.
    while processed < BURST_FRAMES:
        ok, frame = cap.read()
        if not ok:
            break
        processed += 1

        plates = detector.detect_all_plates(frame, conf=DETECT_CONF)
        if not plates:
            misses += 1
            if misses >= BURST_MISS_LIMIT:
                break
            continue
        misses = 0  # reset: a plate was found this frame

        for found in plates:
            plate = found["plate"]
            conf = found["confidence"]
            ocr = found.get("ocr_score", 0.0)
            if plate not in best or ocr > best[plate][1]:
                best[plate] = (conf, ocr)

    # Emit every unique plate once, best OCR score first (order is cosmetic).
    for plate, (conf, ocr) in sorted(best.items(), key=lambda kv: kv[1][1], reverse=True):
        on_plate(plate, conf, ocr)

    return processed


def monitor_sector(sector_id, video_path, on_plate, should_stop):
    """Continuously monitor one sector video until it ends or stop is set.

    Args:
        sector_id: sector name (for logging only).
        video_path: path to the sector's video file.
        on_plate: callback(plate, confidence, ocr_score) per plate read.
        should_stop: callable returning True when the worker should exit.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[cctv] cannot open video for sector {sector_id}: {video_path}")
        return

    subtractor = _new_subtractor()
    frame_idx = 0
    cooldown_frames = 0  # frames remaining before motion can trigger again

    while not should_stop():
        ok, frame = cap.read()
        if not ok:
            break  # end of file (demo). For a live stream, reconnect here.

        frame_idx += 1

        if cooldown_frames > 0:
            cooldown_frames -= 1
            continue  # cooling down after the last burst (counted in frames)

        if frame_idx % FRAME_SKIP != 0:
            continue

        # Feed a downscaled frame to the subtractor for speed.
        small = cv2.resize(frame, (320, 180))
        mask = subtractor.apply(small)

        if _motion_fraction(mask) >= MOTION_THRESHOLD:
            # Motion detected: capture a (smart) burst for multi-angle reads.
            _capture_burst(cap, on_plate)
            cooldown_frames = MOTION_COOLDOWN_FRAMES

    cap.release()
