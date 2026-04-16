#!/usr/bin/env python3
"""
Extract pairwise distance features from detections + annotations for training.

Outputs a CSV with one row per (frame, person_i, person_j) pair:
  frame, person_i, person_j, distance, delta_distance,
  velocity_alignment, dist_std, proximity_streak, same_group

Usage:
    python extract_features.py
    python extract_features.py --frames-dir frames --output features.csv
"""

import argparse
import csv
import json
import math
from collections import deque
from pathlib import Path


PROXIMITY_THRESHOLD = 200  # pixels
WINDOW = 5                 # frames for rolling features


def bbox_center(bbox):
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def velocity_alignment(v1, v2):
    """Cosine similarity of two velocity vectors. Returns 0 if either is zero."""
    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return (v1[0] * v2[0] + v1[1] * v2[1]) / (mag1 * mag2)


def extract(frames_dir: Path, output_path: Path, append: bool = False):
    det_file = frames_dir / "detections_cache.json"
    ann_file = frames_dir / "annotations.json"

    if not det_file.exists():
        raise FileNotFoundError(f"No detections cache found at {det_file}")
    if not ann_file.exists():
        raise FileNotFoundError(f"No annotations found at {ann_file}. Label some frames first.")

    with open(det_file) as f:
        detections = json.load(f)
    with open(ann_file) as f:
        annotations = json.load(f)

    frames = sorted(annotations.keys())
    print(f"Processing {len(frames)} annotated frames...")

    def build_frame_data(frame_name):
        ann = {p["person_id"]: p["group_id"] for p in annotations.get(frame_name, [])}
        det = {p["person_id"]: p["bbox"]    for p in detections.get(frame_name, [])}
        result = {}
        for pid, bbox in det.items():
            group = ann.get(pid)
            if group is None:
                continue
            result[pid] = {"center": bbox_center(bbox), "group_id": group}
        return result

    prev_distances: dict[tuple, float] = {}
    prev_centers: dict[int, tuple] = {}
    dist_history: dict[tuple, deque] = {}
    rows = []

    for frame_name in frames:
        frame_data = build_frame_data(frame_name)
        pids = sorted(frame_data.keys())

        # Compute per-person velocities
        velocities = {}
        for pid, data in frame_data.items():
            cx, cy = data["center"]
            if pid in prev_centers:
                px, py = prev_centers[pid]
                velocities[pid] = (cx - px, cy - py)
            else:
                velocities[pid] = (0.0, 0.0)
            prev_centers[pid] = (cx, cy)

        for a, pid_i in enumerate(pids):
            for pid_j in pids[a + 1:]:
                di = frame_data[pid_i]
                dj = frame_data[pid_j]

                dist = distance(di["center"], dj["center"])
                pair = (min(pid_i, pid_j), max(pid_i, pid_j))

                delta = dist - prev_distances[pair] if pair in prev_distances else 0.0
                prev_distances[pair] = dist

                vel_align = velocity_alignment(velocities[pid_i], velocities[pid_j])

                if pair not in dist_history:
                    dist_history[pair] = deque(maxlen=WINDOW)
                dist_history[pair].append(dist)

                history = dist_history[pair]
                dist_std = (
                    math.sqrt(sum((d - sum(history) / len(history)) ** 2 for d in history) / len(history))
                    if len(history) > 1 else 0.0
                )
                proximity_streak = sum(1 for d in history if d < PROXIMITY_THRESHOLD) / len(history)

                same_group = int(
                    di["group_id"] is not None
                    and dj["group_id"] is not None
                    and di["group_id"] == dj["group_id"]
                )

                rows.append({
                    "frame":              frame_name,
                    "person_i":           pid_i,
                    "person_j":           pid_j,
                    "distance":           round(dist, 2),
                    "delta_distance":     round(delta, 2),
                    "velocity_alignment": round(vel_align, 4),
                    "dist_std":           round(dist_std, 2),
                    "proximity_streak":   round(proximity_streak, 4),
                    "same_group":         same_group,
                })

    fieldnames = [
        "frame", "person_i", "person_j",
        "distance", "delta_distance", "velocity_alignment", "dist_std", "proximity_streak",
        "same_group",
    ]
    mode = "a" if append and output_path.exists() else "w"
    with open(output_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    positives = sum(r["same_group"] for r in rows)
    print(f"Done. {total} pairs written to {output_path}")
    print(f"  same_group=1: {positives}  ({100*positives//total}%)")
    print(f"  same_group=0: {total - positives}  ({100*(total-positives)//total}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", default="frames", help="Directory with detections_cache.json and annotations.json")
    parser.add_argument("--output", default="features.csv", help="Output CSV path")
    parser.add_argument("--append", action="store_true", help="Append to existing CSV instead of overwriting")
    args = parser.parse_args()

    extract(Path(args.frames_dir), Path(args.output), append=args.append)
