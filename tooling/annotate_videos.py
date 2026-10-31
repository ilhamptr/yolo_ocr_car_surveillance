"""Evaluation tool: draw YOLO plate boxes + OCR text onto CCTV videos.

For each sector folder (videos/A1 .. videos/A4), reads the video, runs
YOLO + EasyOCR on every (sampled) frame, draws the bounding box and the
read plate text, and writes an annotated copy to annotated/<sector>.mp4.

This is for visually checking detection/OCR quality, separate from the app.

Usage:
    python annotate_videos.py
    python annotate_videos.py --sectors A1 A2 --conf 0.25 --stride 1
"""

import argparse
from pathlib import Path

import cv2

from app import detector

VIDEOS_DIR = Path("videos")
OUT_DIR = Path("annotated")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


def find_video(sector_dir: Path):
    """Return the first video file inside a sector folder, or None."""
    for f in sorted(sector_dir.iterdir()):
        if f.suffix.lower() in VIDEO_EXTS:
            return f
    return None


def annotate_video(sector_id: str, video_path: Path, conf: float, stride: int):
    """Annotate one video and save it to OUT_DIR/<sector_id>.mp4."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[{sector_id}] cannot open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{sector_id}.mp4"
    # Output fps is divided by stride so playback speed stays natural.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps / stride, (w, h))

    frame_idx = 0
    detections_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride != 0:
            continue

        plates = detector.detect_all_plates(frame, conf=conf, min_chars=1)
        for p in plates:
            x1, y1, x2, y2 = p["box"]
            text = p["plate"] or "?"
            # Show both scores: det = YOLO box, ocr = text recognition.
            label = f"{text} det{p['confidence']:.2f} ocr{p.get('ocr_score', 0):.2f}"
            # Box.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Label background for readability.
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            detections_count += 1

        # Sector banner so it's clear which camera this is.
        cv2.putText(frame, f"Sector {sector_id}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        writer.write(frame)

        if total and frame_idx % (stride * 50) == 0:
            pct = 100 * frame_idx / total
            print(f"[{sector_id}] {pct:.0f}%  ({detections_count} boxes so far)")

    cap.release()
    writer.release()
    print(f"[{sector_id}] done -> {out_path}  ({detections_count} boxes total)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sectors", nargs="*", default=["A1", "A2", "A3", "A4"],
                    help="sector folder names under videos/")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="YOLO confidence threshold")
    ap.add_argument("--stride", type=int, default=3,
                    help="process every Nth frame (higher = faster, choppier)")
    args = ap.parse_args()

    for sector_id in args.sectors:
        sector_dir = VIDEOS_DIR / sector_id
        if not sector_dir.is_dir():
            print(f"[{sector_id}] folder not found: {sector_dir}")
            continue
        video = find_video(sector_dir)
        if video is None:
            print(f"[{sector_id}] no video file in {sector_dir}")
            continue
        annotate_video(sector_id, video, args.conf, args.stride)


if __name__ == "__main__":
    main()
