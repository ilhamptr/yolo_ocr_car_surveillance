# CCTV Parking Zone Tracker

Track where vehicles are parked from CCTV footage. Each sector camera reads the plates of passing cars (YOLO + OCR), and the system stores the **last sector** each plate was seen in. Users simply type their plate in the UI to find which sector their car is parked in.

Core idea: CCTV catches a plate while the car is **moving** past a camera, not while parked. So the last sector a plate was seen in is the most likely place it parked. YOLO only runs when motion is detected, so idle cameras cost almost nothing.

## Architecture

```
Per-sector video
   -> Motion gate (MOG2)        app/cctv.py
   -> Plate detection (YOLO)    app/detector.py
   -> Crop + preprocessing
   -> OCR (PaddleOCR/EasyOCR)   app/detector.py
   -> Plate character fixing    app/detector.py
   -> Store (SQLite)            app/database.py
   -> API (FastAPI)             app/api.py
   -> UI (Streamlit)            app/ui.py
```

One background thread per sector runs continuously (managed by `app/scheduler.py`). When MOG2 detects motion, the system runs a *burst* of YOLO + OCR over several frames to read a plate from multiple angles, then keeps the best result. A separate timer thread removes plates not seen for over 24 hours.

## Runtime structure

| File | Role |
|------|------|
| `app/cctv.py` | Read per-sector video, motion gate (MOG2), burst capture |
| `app/detector.py` | YOLO plate detection + crop preprocessing + OCR + Indonesian plate correction |
| `app/scheduler.py` | One worker thread per sector + expiry timer |
| `app/database.py` | SQLite, one row per plate (parking snapshot, not a log) |
| `app/api.py` | FastAPI endpoints (`/search`, `/vehicle`, `/vehicles`, `/status`) |
| `app/ui.py` | Streamlit UI to look up a plate |
| `best.pt` | Trained YOLO model (used at runtime) |

## Video layout

One subfolder per sector inside `videos/`:

```
videos/
  A1/ cam.mp4
  A2/ cam.mp4
  A3/ cam.mp4
  A4/ cam.mp4
```

## Running

### Local (no Docker)

```bash
python -m venv env
source env/bin/activate
pip install -r requirements_app.txt
./run_local.sh
```

Open http://127.0.0.1:8501 for the UI. The API runs at http://127.0.0.1:8000.

### Docker

```bash
docker compose up --build
```

- UI: http://localhost:8501
- API: http://localhost:8000

Put your CCTV videos in `./videos/<sector>/` (mounted read-only into the container).

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/search?plate=` | Find parked vehicles by (partial) plate |
| GET | `/vehicle/{plate}` | Exact lookup: which sector this plate is in now |
| GET | `/vehicles` | List every currently-parked vehicle |
| GET | `/status` | Worker stats + sectors being monitored |

## Configuration (env vars)

All optional with sensible defaults. See `docker-compose.yml` for example values.

| Var | Default | Description |
|-----|---------|-------------|
| `VIDEOS_DIR` | `./videos` | Sector videos folder |
| `DB_PATH` | `./parking.db` | SQLite database location |
| `OCR_ENGINE` | `paddle` | `paddle` or `easyocr` |
| `DETECT_CONF` | `0.25` | YOLO confidence threshold |
| `MIN_OCR_SCORE` | `0.65` | Drop OCR readings below this score |
| `PLATE_CORRECTION` | `1` | Fix look-alike characters using Indonesian plate structure |
| `MOTION_THRESHOLD` | `0.02` | Fraction of frame moving to trigger YOLO |
| `FRAME_SKIP` | `5` | Check motion every Kth frame |
| `BURST_FRAMES` | `12` | Max frames read per motion event |
| `EXPIRY_HOURS` | `24` | Remove a plate not seen for this many hours |

## Tooling (outside runtime)

The `tooling/` folder holds training/evaluation scripts, not needed to run the app:

- `tooling/main.py` — train YOLOv8n from `tooling/data.yaml` (produces `best.pt`)
- `tooling/conv_dataset.py` — convert an NDJSON dataset to YOLO format
- `tooling/annotate_videos.py` — draw boxes + OCR text onto videos to visually check quality
- `tooling/visualize_errors.py` — visualize model FP/FN on the test set
- `tooling/evaluate_test.ipynb` — evaluation notebook

The `misc/` folder holds local data not pushed to the repo (datasets, raw NDJSON, the runtime `parking.db`, and the virtualenv). It is git-ignored.
