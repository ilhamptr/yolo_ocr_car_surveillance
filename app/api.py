"""FastAPI backend for the CCTV parking-zone tracker.

Background workers (one per sector) watch each CCTV video continuously and,
when motion is detected, read plates and update the database. The database
is a live snapshot of what is parked where. Users only search.

Endpoints:
  GET  /search?plate=    find currently-parked vehicles by (partial) plate
  GET  /vehicle/{plate}  exact lookup: which sector is this plate in now
  GET  /vehicles         list everything currently parked
  GET  /status           worker stats + sectors being monitored
"""

from fastapi import FastAPI, HTTPException

from app import database, detector, scheduler

app = FastAPI(title="CCTV Parking Zone Tracker")


@app.on_event("startup")
def startup():
    database.init_db()
    # Load both models once, up front, so the per-sector worker threads don't
    # initialise PaddleOCR concurrently (which can segfault).
    detector.warm_up()
    scheduler.start()


@app.on_event("shutdown")
def shutdown():
    scheduler.stop()


@app.get("/search")
def search(plate: str):
    """Find currently-parked vehicles whose plate contains the query."""
    return {"query": plate, "results": database.search_plate(plate)}


@app.get("/vehicle/{plate}")
def vehicle(plate: str):
    """Exact lookup: return the sector where this plate is currently parked."""
    record = database.find_plate(plate)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Plate '{plate}' is not currently parked.",
        )
    return record


@app.get("/vehicles")
def vehicles():
    """List every currently-parked vehicle."""
    return {"results": database.list_all()}


@app.get("/status")
def status():
    """Return worker stats and the sectors being monitored."""
    return scheduler.stats()
