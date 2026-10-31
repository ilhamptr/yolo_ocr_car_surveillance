"""License plate detection (YOLO) + OCR (EasyOCR).

Pipeline for a single frame:
  1. YOLO locates every plate and returns bounding boxes.
  2. Each box is cropped from the frame.
  3. EasyOCR reads the text inside each crop.
"""

import os
import re
import threading
from pathlib import Path

import cv2
import numpy as np

# Heavy models are loaded lazily so importing this module stays cheap.
_yolo = None
_reader = None

# PaddlePaddle / PaddleOCR is not thread-safe for concurrent init or inference.
# With one worker thread per sector, parallel calls segfault. These locks
# serialise model creation and each OCR/YOLO inference call.
_reader_init_lock = threading.Lock()
_infer_lock = threading.Lock()

MODEL_PATH = Path(__file__).parent.parent / "best.pt"

# --- crop / preprocessing tunables (env-configurable) ---
# Expand each YOLO box by this fraction of its size before cropping, so a
# slightly-tight box still captures the whole plate. 0.15 = 15% padding.
CROP_PADDING = float(os.environ.get("CROP_PADDING", 0.15))
# Upscale small crops so their height reaches at least this many pixels.
# Low-res CCTV plates read much better after upscaling.
MIN_CROP_HEIGHT = int(os.environ.get("MIN_CROP_HEIGHT", 64))

# Which OCR engine to use: "paddle" (default, more accurate) or "easyocr".
OCR_ENGINE = os.environ.get("OCR_ENGINE", "paddle").lower()

# Correct look-alike characters using the Indonesian plate structure.
PLATE_CORRECTION = os.environ.get("PLATE_CORRECTION", "1") not in ("0", "false", "False")

# Minimum OCR confidence (0-1) for a reading to be kept. Readings below this
# are likely garbage, not "slightly wrong", so we drop them. Set to 0 to keep
# everything. This measures how confident OCR is in the TEXT, not YOLO's box.
MIN_OCR_SCORE = float(os.environ.get("MIN_OCR_SCORE", 0.65))


def _get_yolo():
    global _yolo
    if _yolo is None:
        from ultralytics import YOLO
        _yolo = YOLO(str(MODEL_PATH))
    return _yolo


def _get_reader():
    """Lazily build the OCR reader. Thread-safe: only one thread builds it."""
    global _reader
    if _reader is not None:
        return _reader

    with _reader_init_lock:
        # Re-check inside the lock in case another thread just built it.
        if _reader is not None:
            return _reader

        if OCR_ENGINE == "easyocr":
            import easyocr
            model_dir = os.environ.get("EASYOCR_MODULE_PATH")
            kwargs = {"model_storage_directory": model_dir} if model_dir else {}
            reader = easyocr.Reader(["en"], gpu=False, **kwargs)
        else:
            reader = _build_paddle()
        _reader = reader

    return _reader


def _paddle_major_version() -> int:
    """Return PaddleOCR's major version (2 or 3). Defaults to 3 if unknown."""
    try:
        import paddleocr
        return int(str(paddleocr.__version__).split(".")[0])
    except Exception:
        return 3


def _build_paddle():
    """Construct a PaddleOCR reader for whichever major version is installed.

    The 2.x and 3.x constructors take different arguments:
      - 3.x: use_textline_orientation, use_doc_orientation_classify, ...
      - 2.x: use_angle_cls, show_log
    Docker pins 2.10.0 (stable on ARM64 Linux); local may run 3.x. This keeps
    both working without code changes per environment.
    """
    from paddleocr import PaddleOCR

    if _paddle_major_version() >= 3:
        return PaddleOCR(
            lang="en",
            use_textline_orientation=True,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
    # 2.x API
    return PaddleOCR(lang="en", use_angle_cls=True, show_log=False)


def warm_up():
    """Eagerly load both models once, before worker threads start.

    This avoids several threads initialising PaddleOCR at the same moment,
    which can segfault. Safe to call multiple times.
    """
    _get_yolo()
    _get_reader()


def _clean_plate_text(text: str) -> str:
    """Keep only A-Z and 0-9, uppercase. Drops spaces and punctuation."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


# Ambiguous OCR character pairs, resolved by position in the plate.
# In a numeric position, a letter that looks like a digit becomes that digit.
_TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
             "Z": "2", "B": "8", "S": "5", "G": "6", "T": "7", "A": "4"}
# In a letter position, a digit that looks like a letter becomes that letter.
_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "8": "B", "5": "S", "6": "G", "4": "A"}

def _to_digits(s):
    return "".join(_TO_DIGIT.get(c, c) for c in s)


def _to_letters(s):
    return "".join(_TO_LETTER.get(c, c) for c in s)


def _correct_plate(text: str) -> str:
    """Fix look-alike OCR characters using the Indonesian plate structure.

    An Indonesian plate is: 1-2 area LETTERS, 1-4 DIGITS, 1-3 series LETTERS,
    e.g. B 1234 XYZ. We locate the three zones by the position of the digit
    block, then coerce each character toward the type expected in its zone
    (letter-like digits -> letters at the ends, digit-like letters -> digits
    in the middle).

    Zones are found by first normalising obvious types: the leading run that
    is letter-dominant is the area, the trailing letter-dominant run is the
    series, and the middle is the number. If the overall length is outside
    the valid 3-9 range, the text is returned unchanged.
    """
    if not (3 <= len(text) <= 9):
        return text

    n = len(text)
    # Zone boundaries are decided from the ACTUAL character types (alpha vs
    # digit), not the look-alike maps. The maps are only used afterwards to
    # coerce characters within a zone. This avoids a digit like '1' being
    # mistaken for a letter when locating the area/series zones.
    area_len = 2 if n >= 5 and text[0].isalpha() and text[1].isalpha() else 1

    # Series: up to 3 trailing alphabetic chars (at least 1).
    series_len = 0
    for c in reversed(text):
        if c.isalpha() and series_len < 3:
            series_len += 1
        else:
            break
    series_len = max(1, series_len)

    # Ensure at least one digit remains in the middle.
    if area_len + series_len >= n:
        series_len = max(1, n - area_len - 1)

    area = text[:area_len]
    series = text[n - series_len:]
    number = text[area_len:n - series_len]

    if not number:
        return text  # no number zone; leave as-is

    return _to_letters(area) + _to_digits(number) + _to_letters(series)


def _preprocess_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """Clean up a plate crop before OCR.

    Steps, in order:
      1. Upscale small crops (low-res CCTV plates) to a readable height.
      2. Convert to grayscale.
      3. CLAHE to boost local contrast (handles uneven lighting).
      4. Unsharp mask to sharpen edges blurred by low video quality.

    Returns a BGR image (3 channels) so EasyOCR's RGB path still works.
    """
    img = crop_bgr

    # 1. Upscale if the crop is short. Cubic keeps edges smoother than linear.
    h = img.shape[0]
    if 0 < h < MIN_CROP_HEIGHT:
        scale = MIN_CROP_HEIGHT / h
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. CLAHE: local contrast enhancement.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # 4. Unsharp mask: sharpen by subtracting a blurred version.
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
    sharp = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    # Back to 3-channel BGR for a consistent downstream path.
    return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)


def _read_easyocr(reader, crop_bgr):
    """OCR with EasyOCR. Returns (raw_text, mean_score).

    EasyOCR gives a confidence per fragment; we average them for the plate.
    """
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    with _infer_lock:
        results = reader.readtext(crop_rgb)
    if not results:
        return "", 0.0
    results.sort(key=lambda r: r[0][0][0])  # by x position
    text = "".join(r[1] for r in results)
    scores = [float(r[2]) for r in results]
    return text, sum(scores) / len(scores)


def _read_paddle(reader, crop_bgr):
    """OCR with PaddleOCR. Returns (raw_text, mean_score).

    Handles both major versions, which return different structures:
      - 3.x: reader.predict() -> [OCRResult dict] with rec_texts/rec_scores/rec_boxes
      - 2.x: reader.ocr(img, cls=True) -> [[ [box, (text, score)], ... ]]
    Fragments are sorted left-to-right; scores are averaged.
    """
    if _paddle_major_version() >= 3:
        with _infer_lock:
            result = reader.predict(crop_bgr)
        if not result:
            return "", 0.0
        res = result[0]
        texts = list(res.get("rec_texts") or [])
        scores = list(res.get("rec_scores") or [])
        boxes = res.get("rec_boxes")
        boxes = list(boxes) if boxes is not None else []
        if not texts:
            return "", 0.0
        if len(boxes) == len(texts) and len(texts) > 1:
            order = sorted(range(len(texts)), key=lambda i: boxes[i][0])
            texts = [texts[i] for i in order]
            if len(scores) == len(order):
                scores = [scores[i] for i in order]
        mean_score = sum(float(s) for s in scores) / len(scores) if scores else 0.0
        return "".join(texts), mean_score

    # --- 2.x path ---
    with _infer_lock:
        result = reader.ocr(crop_bgr, cls=True)
    # result: [ [ [box_points, (text, score)], ... ] ]  (None/empty if nothing)
    if not result or not result[0]:
        return "", 0.0
    lines = result[0]
    # Each line: [ [[x,y],...4 pts], (text, score) ]; sort by left x.
    lines.sort(key=lambda ln: ln[0][0][0])
    texts = [ln[1][0] for ln in lines]
    scores = [float(ln[1][1]) for ln in lines]
    mean_score = sum(scores) / len(scores) if scores else 0.0
    return "".join(texts), mean_score


def read_text(crop_bgr: np.ndarray, preprocess: bool = True):
    """Run OCR on a cropped plate image.

    Returns (plate_text, ocr_score) where ocr_score is 0-1. Empty input
    returns ("", 0.0).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "", 0.0

    if preprocess:
        crop_bgr = _preprocess_crop(crop_bgr)

    reader = _get_reader()
    if OCR_ENGINE == "easyocr":
        raw, score = _read_easyocr(reader, crop_bgr)
    else:
        raw, score = _read_paddle(reader, crop_bgr)

    cleaned = _clean_plate_text(raw)
    if PLATE_CORRECTION:
        cleaned = _correct_plate(cleaned)
    return cleaned, score


def detect_all_plates(image_bgr: np.ndarray, conf: float = 0.25,
                      min_chars: int = 3, max_chars: int = 9):
    """Detect and read EVERY plate in a frame.

    Args:
        image_bgr: frame as a BGR numpy array (OpenCV format).
        conf: YOLO confidence threshold.
        min_chars: discard OCR results shorter than this (likely noise).
        max_chars: discard results longer than this. An Indonesian plate is at
                   most 2+4+3 = 9 characters, so longer reads are banners/signs.

    Returns:
        list of dicts: {plate, confidence, ocr_score, box}.
        confidence is YOLO's box score; ocr_score is OCR's text score.
        Entries outside the length range or below MIN_OCR_SCORE are dropped.
    """
    yolo = _get_yolo()
    # Serialise inference across threads: Paddle/torch backends here are not
    # safe to call concurrently from multiple sector workers.
    with _infer_lock:
        result = yolo.predict(image_bgr, conf=conf, verbose=False)[0]

    if len(result.boxes) == 0:
        return []

    h, w = image_bgr.shape[:2]
    boxes = result.boxes.xyxy.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    plates = []
    for box, det_conf in zip(boxes, confidences):
        x1, y1, x2, y2 = map(int, box)

        # Expand the box by CROP_PADDING so a slightly-tight detection still
        # captures the whole plate. Padding scales with box size.
        bw, bh = x2 - x1, y2 - y1
        px, py = int(bw * CROP_PADDING), int(bh * CROP_PADDING)
        cx1, cy1 = x1 - px, y1 - py
        cx2, cy2 = x2 + px, y2 + py

        # Clamp the padded crop to image bounds.
        cx1, cy1 = max(0, cx1), max(0, cy1)
        cx2, cy2 = min(w, cx2), min(h, cy2)
        crop = image_bgr[cy1:cy2, cx1:cx2]

        text, ocr_score = read_text(crop)
        if not (min_chars <= len(text) <= max_chars):
            continue  # too short (noise) or too long (banner/sign, not a plate)
        if ocr_score < MIN_OCR_SCORE:
            continue  # too low confidence in the TEXT -> likely garbage

        plates.append({
            "plate": text,
            "confidence": float(det_conf),
            "ocr_score": float(ocr_score),
            # Report the padded crop box so annotations show what OCR saw.
            "box": [cx1, cy1, cx2, cy2],
        })

    return plates
