"""
Social Group Labeler — FastAPI backend

Usage:
    FRAMES_DIR=/path/to/frames uvicorn main:app --reload

The first run will execute SAM3 detection on all frames and cache the results
to <FRAMES_DIR>/detections_cache.json. Subsequent runs load the cache instantly.
"""

import json
import os
import threading
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── Config ────────────────────────────────────────────────────────────────────

FRAMES_DIR = Path(os.environ.get("FRAMES_DIR", "frames"))
CACHE_FILE = FRAMES_DIR / "detections_cache.json"
ANNOTATIONS_FILE = FRAMES_DIR / "annotations.json"

# ─── Shared state ──────────────────────────────────────────────────────────────

_status: dict[str, Any] = {"state": "idle", "progress": 0, "total": 0, "error": None}
_detections: dict[str, list] = {}

app = FastAPI()

# ─── Detection pipeline ────────────────────────────────────────────────────────


def _extract_persons(results, frame_idx: int) -> list[dict]:
    """Pull bboxes, track IDs, and masks out of an ultralytics Results list."""
    persons = []
    for r in results:
        if r.boxes is None:
            continue
        boxes = r.boxes
        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[j].tolist()
            track_id = (
                int(boxes.id[j].item())
                if boxes.id is not None
                else frame_idx * 10000 + j
            )
            conf = float(boxes.conf[j].item()) if boxes.conf is not None else 0.0
            person: dict[str, Any] = {
                "person_id": track_id,
                "bbox": [round(x1), round(y1), round(x2 - x1), round(y2 - y1)],
                "confidence": round(conf, 3),
            }
            if r.masks is not None and j < len(r.masks.xy):
                person["mask"] = r.masks.xy[j].tolist()
            persons.append(person)
    return persons


def _run_sam3(frames: list[Path]) -> dict[str, list]:
    """
    Run SAM3 person detection and tracking on a sequence of frames.

    SAM3 is prompted with the text "person" on the first frame to auto-detect
    all individuals, then uses built-in tracking (persist=True) to maintain
    consistent person IDs across subsequent frames.
    """
    from ultralytics import SAM  # pip install ultralytics

    model = None
    for name in ("sam3.1_b.pt", "sam3_b.pt", "sam3.pt"):
        try:
            model = SAM(name)
            print(f"Loaded SAM3 model: {name}")
            break
        except Exception:
            continue
    if model is None:
        raise RuntimeError(
            "Could not load a SAM3 model. Make sure a SAM3 checkpoint is available "
            "(e.g. sam3.1_b.pt) and ultralytics is up to date."
        )

    results_dict: dict[str, list] = {}
    for i, frame_path in enumerate(frames):
        _status["progress"] = i + 1
        if i == 0:
            results = model.predict(str(frame_path), texts=["person"], conf=0.3, verbose=False)
        else:
            results = model.track(
                str(frame_path), texts=["person"], persist=True, conf=0.3, verbose=False
            )
        results_dict[frame_path.name] = _extract_persons(results, i)

    return results_dict


def _run_yolo_fallback(frames: list[Path]) -> dict[str, list]:
    """
    Fallback: use YOLO11 with ByteTrack to detect and track persons.
    Yields bounding boxes and consistent person IDs; no segmentation masks.
    """
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")
    print("Using YOLO11 fallback for detection.")

    results_dict: dict[str, list] = {}
    for i, frame_path in enumerate(frames):
        _status["progress"] = i + 1
        if i == 0:
            results = model.predict(
                str(frame_path), classes=[0], conf=0.3, verbose=False
            )
        else:
            results = model.track(
                str(frame_path), classes=[0], persist=True, conf=0.3, verbose=False
            )
        results_dict[frame_path.name] = _extract_persons(results, i)

    return results_dict


def _detection_worker():
    global _detections, _status

    frames = sorted(p for ext in ("*.png", "*.jpg", "*.jpeg") for p in FRAMES_DIR.glob(ext))
    if not frames:
        _status["state"] = "no_frames"
        print(f"No image frames found in {FRAMES_DIR.resolve()}")
        return

    _status["state"] = "running"
    _status["total"] = len(frames)
    print(f"Running detection on {len(frames)} frames in {FRAMES_DIR.resolve()} ...")

    try:
        _detections = _run_sam3(frames)
    except Exception as e:
        print(f"SAM3 failed: {e}\nFalling back to YOLO11 ...")
        try:
            _detections = _run_yolo_fallback(frames)
        except Exception as e2:
            _status["state"] = "error"
            _status["error"] = str(e2)
            print(f"Fallback also failed: {e2}")
            return

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w") as f:
        json.dump(_detections, f)

    _status["state"] = "done"
    print(f"Detection complete. Cache saved to {CACHE_FILE}")


# ─── Lifecycle ─────────────────────────────────────────────────────────────────


@app.on_event("startup")
async def startup():
    global _detections
    if not FRAMES_DIR.exists():
        print(
            f"[warn] FRAMES_DIR '{FRAMES_DIR.resolve()}' does not exist. "
            "Set the FRAMES_DIR environment variable to your frames folder."
        )
        _status["state"] = "no_frames"
        return

    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            _detections = json.load(f)
        _status["state"] = "done"
        print(f"Loaded detection cache from {CACHE_FILE} ({len(_detections)} frames)")
    else:
        t = threading.Thread(target=_detection_worker, daemon=True)
        t.start()


# ─── API ───────────────────────────────────────────────────────────────────────


@app.get("/api/status")
def get_status():
    return _status


@app.get("/api/frames")
def get_frames():
    if not FRAMES_DIR.exists():
        return {"frames": []}
    frames = sorted(f.name for ext in ("*.png", "*.jpg", "*.jpeg") for f in FRAMES_DIR.glob(ext))
    return {"frames": frames}


@app.get("/api/frame/{frame_name:path}")
def get_frame(frame_name: str):
    path = FRAMES_DIR / frame_name
    if not path.exists() or not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    media_type = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    return FileResponse(str(path), media_type=media_type)


@app.get("/api/detections")
def get_detections():
    return _detections


@app.get("/api/annotations")
def get_annotations():
    if ANNOTATIONS_FILE.exists():
        with open(ANNOTATIONS_FILE) as f:
            return json.load(f)
    return {}


class AnnotationPayload(BaseModel):
    annotations: dict


@app.post("/api/annotations")
def save_annotations(payload: AnnotationPayload):
    ANNOTATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ANNOTATIONS_FILE, "w") as f:
        json.dump(payload.annotations, f, indent=2)
    return {"status": "saved", "path": str(ANNOTATIONS_FILE.resolve())}


# Serve the frontend — must be registered last
app.mount("/", StaticFiles(directory="static", html=True), name="static")

# ─── Dev entrypoint ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
