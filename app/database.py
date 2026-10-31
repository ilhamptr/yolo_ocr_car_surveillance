"""SQLite storage for current parking state.

The table holds ONE row per plate: its current sector and when it was last
seen. It is a live snapshot of the parking lot, not a history log.

CCTV catches a plate while the car is MOVING past a camera, not while it is
parked. So the last sector a plate was seen in is the most probable place it
parked. Detection is continuous and motion-gated: whenever a camera sees
movement it reads plates and calls record_detection() for each one.

Rules:
  - plate seen in a new/different sector -> update to the new sector
  - plate seen in the same sector        -> refresh last_seen
  - plate not seen for a while           -> kept (a missed read != car left)
  - plate last_seen older than 24 hours  -> removed by expire_old()
"""

import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

# A plate not seen for longer than this is considered gone.
EXPIRY_HOURS = int(os.environ.get("EXPIRY_HOURS", 24))

DB_PATH = Path(os.environ.get("DB_PATH", Path(__file__).parent.parent / "parking.db"))


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the table if it does not exist yet."""
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicles (
                plate       TEXT PRIMARY KEY,
                sector      TEXT NOT NULL,
                confidence  REAL,   -- YOLO detection score
                ocr_score   REAL,   -- OCR text-recognition score
                last_seen   TEXT NOT NULL
            )
            """
        )


def record_detection(plate: str, sector: str, confidence: float = None,
                      ocr_score: float = None):
    """Record a single plate reading.

    Upserts the plate to the given sector with a fresh timestamp. The latest
    sector a plate is seen in always wins (handles a car moving between
    cameras). confidence is YOLO's box score; ocr_score is the text score.
    Returns "added" or "updated".
    """
    now_iso = datetime.now().isoformat(timespec="seconds")
    plate = plate.upper()

    with get_conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM vehicles WHERE plate = ?", (plate,)
        ).fetchone()
        conn.execute(
            """
            INSERT INTO vehicles (plate, sector, confidence, ocr_score, last_seen)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(plate) DO UPDATE SET
                sector = excluded.sector,
                confidence = excluded.confidence,
                ocr_score = excluded.ocr_score,
                last_seen = excluded.last_seen
            """,
            (plate, sector, confidence, ocr_score, now_iso),
        )
    return "updated" if exists else "added"


def expire_old():
    """Delete plates not seen for longer than EXPIRY_HOURS. Returns count removed."""
    cutoff = (datetime.now() - timedelta(hours=EXPIRY_HOURS)).isoformat(
        timespec="seconds"
    )
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM vehicles WHERE last_seen < ?", (cutoff,))
        return cur.rowcount


def find_plate(plate: str):
    """Return the current record for an exact plate, or None if not parked."""
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM vehicles WHERE plate = ?", (plate.upper(),)
        ).fetchone()
        return dict(row) if row else None


def search_plate(query: str):
    """Return all currently-parked vehicles whose plate contains the query."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM vehicles WHERE plate LIKE ? ORDER BY sector, plate",
            (f"%{query.upper()}%",),
        ).fetchall()
        return [dict(r) for r in rows]


def list_all():
    """Return every currently-parked vehicle."""
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM vehicles ORDER BY sector, plate"
        ).fetchall()
        return [dict(r) for r in rows]


if __name__ == "__main__":
    init_db()
    print(f"Database ready: {DB_PATH}")
