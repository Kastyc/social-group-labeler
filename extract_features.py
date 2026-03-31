#!/usr/bin/env python3
"""
Extract pairwise distance features from detections + annotations for training.

Outputs a CSV with one row per (frame, person_i, person_j) pair:
  frame, person_i, person_j, distance, delta_distance, same_group

Usage:
    python extract_features.py
    python extract_features.py --frames-dir frames --output features.csv
"""

import argparse
import csv
import json
import math
from pathlib import Path


def bbox_center(bbox):
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def extract(frames_dir: Path, output_path: Path):
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

    # Only process frames that have been annotated
    frames = sorted(annotations.keys())
    print(f"Processing {len(frames)} annotated frames...")

    # Build lookup: frame -> {person_id: {center, group_id}}
    def build_frame_data(frame_name):
        ann = {p["person_id"]: p["group_id"] for p in annotations.get(frame_name, [])}
        det = {p["person_id"]: p["bbox"]    for p in detections.get(frame_name, [])}
        result = {}
        for pid, bbox in det.items():
            group = ann.get(pid)
            if group is None:
                continue  # skip unlabeled persons
            result[pid] = {"center": bbox_center(bbox), "group_id": group}
        return result

    # prev_distances[frame_name][(i,j)] = distance, for computing delta
    prev_distances: dict[tuple, float] = {}
    rows = []

    for frame_name in frames:
        frame_data = build_frame_data(frame_name)
        pids = sorted(frame_data.keys())

        for a, pid_i in enumerate(pids):
            for pid_j in pids[a + 1:]:
                di = frame_data[pid_i]
                dj = frame_data[pid_j]

                dist = distance(di["center"], dj["center"])
                pair = (min(pid_i, pid_j), max(pid_i, pid_j))
                delta = dist - prev_distances[pair] if pair in prev_distances else 0.0
                prev_distances[pair] = dist

                same_group = int(
                    di["group_id"] is not None
                    and dj["group_id"] is not None
                    and di["group_id"] == dj["group_id"]
                )

                rows.append({
                    "frame":         frame_name,
                    "person_i":      pid_i,
                    "person_j":      pid_j,
                    "distance":      round(dist, 2),
                    "delta_distance": round(delta, 2),
                    "same_group":    same_group,
                })

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "person_i", "person_j",
                                               "distance", "delta_distance", "same_group"])
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
    args = parser.parse_args()

    extract(Path(args.frames_dir), Path(args.output))
