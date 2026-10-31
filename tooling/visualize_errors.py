"""
Pakai model PyTorch (best.pt) untuk prediksi test set,
lalu visualisasikan gambar yang error (deteksi meleset).

Error = False Negative (plat asli ke-miss) atau False Positive (deteksi salah).
Hijau = ground truth (label asli). Merah = prediksi model.
"""

import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---------- SETTING ----------
MODEL_PATH = "./best.pt"  # ganti ke path best.pt kamu
DATA_ROOT = Path("datasets/number-plate-5f0ee7ed")
IMG_DIR = DATA_ROOT / "images" / "test"
LBL_DIR = DATA_ROOT / "labels" / "test"

CONF = 0.20        # confidence threshold prediksi
IOU_MATCH = 0.5    # IoU minimal agar prediksi dianggap "benar" cocok ke GT
MAX_SHOW = 20      # maksimal gambar error yang ditampilkan
# -----------------------------


def yolo_to_xyxy(box, w, h):
    """Konversi label YOLO (cx, cy, bw, bh ternormalisasi) ke piksel (x1,y1,x2,y2)."""
    cx, cy, bw, bh = box
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return [x1, y1, x2, y2]


def iou(a, b):
    """IoU antara dua box format [x1,y1,x2,y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_path, w, h):
    """Baca label YOLO -> list box piksel."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        boxes.append(yolo_to_xyxy((cx, cy, bw, bh), w, h))
    return boxes


def analyze(gt_boxes, pred_boxes):
    """Hitung TP / FP / FN dengan matching IoU."""
    matched_gt = set()
    tp, fp = 0, 0
    for pb in pred_boxes:
        best_iou, best_j = 0.0, -1
        for j, gb in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            cur = iou(pb, gb)
            if cur > best_iou:
                best_iou, best_j = cur, j
        if best_iou >= IOU_MATCH:
            tp += 1
            matched_gt.add(best_j)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def main():
    model = YOLO(MODEL_PATH)

    img_paths = sorted(
        glob.glob(str(IMG_DIR / "*.jpg")) + glob.glob(str(IMG_DIR / "*.png"))
    )
    if not img_paths:
        print(f"Tidak ada gambar di {IMG_DIR}")
        return

    error_images = []  # (path, gt_boxes, pred_boxes, tp, fp, fn)

    print(f"Memeriksa {len(img_paths)} gambar test...")
    for p in img_paths:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes = load_gt(LBL_DIR / (Path(p).stem + ".txt"), w, h)

        res = model.predict(p, conf=CONF, verbose=False)[0]
        pred_boxes = [b.tolist() for b in res.boxes.xyxy.cpu()]

        tp, fp, fn = analyze(gt_boxes, pred_boxes)
        if fp > 0 or fn > 0:  # ada error
            error_images.append((p, gt_boxes, pred_boxes, tp, fp, fn))

    print(f"Ditemukan {len(error_images)} gambar error dari {len(img_paths)} total.")
    if not error_images:
        print("Tidak ada error. Model sempurna di test set ini.")
        return

    # ---------- visualisasi ----------
    show = error_images[:MAX_SHOW]
    cols = 4
    rows = (len(show) + cols - 1) // cols
    plt.figure(figsize=(cols * 4, rows * 4))

    for i, (p, gt, pred, tp, fp, fn) in enumerate(show):
        img = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)

        for b in gt:  # ground truth = hijau
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        for b in pred:  # prediksi = merah
            x1, y1, x2, y2 = map(int, b)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(f"FP={fp} FN={fn}", fontsize=9)
        ax.axis("off")

    plt.suptitle("Hijau = label asli | Merah = prediksi model", fontsize=12)
    plt.tight_layout()
    plt.savefig("test_errors.png", dpi=120, bbox_inches="tight")
    print("Tersimpan: test_errors.png")
    plt.show()


if __name__ == "__main__":
    main()
