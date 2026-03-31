#!/usr/bin/env python3
"""
Train a social group classifier and visualize predictions on frames.

Usage:
    # Train and evaluate
    python train_and_visualize.py --train

    # Run on a specific frame and save visualization
    python train_and_visualize.py --predict --frame frames/logitech_brio_image_raw_000050.png

    # Run on all frames and save to a directory
    python train_and_visualize.py --predict-all --output-dir predictions/
"""

import argparse
import json
import math
import pickle
from pathlib import Path

import cv2
import numpy as np


FEATURES_CSV   = "features.csv"
MODEL_PATH     = "group_classifier.pkl"
DETECTIONS     = "frames/detections_cache.json"
PALETTE = [
    (117, 117, 117),  # grey  — unassigned
    ( 57,  87, 229),  # blue
    ( 67, 160,  71),  # green
    (251, 140,   0),  # orange
    (142,  36, 170),  # purple
    (  0, 172, 193),  # cyan
    (229,  57,  53),  # red
    (  0, 137, 123),  # teal
    (193,  82,  28),  # brown
]


def color(gid):
    if gid is None or gid < 1:
        return PALETTE[0]
    return PALETTE[gid % (len(PALETTE) - 1) + 1]


def bbox_center(bbox):
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def pairwise_features(persons: list[dict], prev_dists: dict) -> tuple[np.ndarray, list]:
    """Return feature matrix and list of (pid_i, pid_j) pairs."""
    rows, pairs = [], []
    pids = [p["person_id"] for p in persons]
    centers = {p["person_id"]: bbox_center(p["bbox"]) for p in persons}

    for a in range(len(pids)):
        for b in range(a + 1, len(pids)):
            i, j = pids[a], pids[b]
            ci, cj = centers[i], centers[j]
            dist = math.sqrt((ci[0] - cj[0]) ** 2 + (ci[1] - cj[1]) ** 2)
            key = (min(i, j), max(i, j))
            delta = dist - prev_dists.get(key, dist)
            prev_dists[key] = dist
            rows.append([dist, delta])
            pairs.append((i, j))

    return np.array(rows) if rows else np.empty((0, 2)), pairs


def connected_components(pids: list, same_group_pairs: list[tuple]) -> dict:
    """Assign group IDs via union-find."""
    parent = {p: p for p in pids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i, j in same_group_pairs:
        union(i, j)

    # Map root → group_id
    roots = {}
    gid = 1
    result = {}
    for p in pids:
        r = find(p)
        if r not in roots:
            roots[r] = gid
            gid += 1
        result[p] = roots[r]

    # Solo people (not in any same-group pair) stay ungrouped
    grouped_pids = {p for pair in same_group_pairs for p in pair}
    for p in pids:
        if p not in grouped_pids:
            result[p] = None

    return result


def draw_predictions(image_path: str, persons: list[dict], group_map: dict) -> np.ndarray:
    img = cv2.imread(image_path)
    for p in persons:
        pid = p["person_id"]
        gid = group_map.get(pid)
        c = color(gid)
        x, y, w, h = [int(v) for v in p["bbox"]]

        cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)

        label = f"G{gid}" if gid else "?"
        cv2.rectangle(img, (x, y - 16), (x + len(label) * 8 + 4, y), c, -1)
        cv2.putText(img, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return img


# ── Train ────────────────────────────────────────────────────────────────────


def train():
    import csv
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    rows = []
    with open(FEATURES_CSV) as f:
        for row in csv.DictReader(f):
            rows.append([float(row["distance"]), float(row["delta_distance"]), int(row["same_group"])])

    if not rows:
        print("features.csv is empty — run extract_features.py first.")
        return

    data = np.array(rows)
    X, y = data[:, :2], data[:, 2]
    print(f"Dataset: {len(X)} pairs  |  positives: {int(y.sum())}  negatives: {int((1-y).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nTest set results:")
    print(classification_report(y_test, y_pred, target_names=["diff group", "same group"]))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    print(f"Model saved to {MODEL_PATH}")


# ── Predict ──────────────────────────────────────────────────────────────────


def predict_frame(image_path: str, clf, detections: dict, prev_dists: dict) -> np.ndarray:
    name = Path(image_path).name
    persons = detections.get(name, [])

    if not persons:
        print(f"No detections for {name}")
        return cv2.imread(image_path)

    X, pairs = pairwise_features(persons, prev_dists)

    if len(X) == 0:
        group_map = {p["person_id"]: 1 for p in persons}
    else:
        preds = clf.predict(X)
        same = [pair for pair, pred in zip(pairs, preds) if pred == 1]
        pids = [p["person_id"] for p in persons]
        group_map = connected_components(pids, same)

    return draw_predictions(image_path, persons, group_map)


def predict_one(image_path: str):
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(DETECTIONS) as f:
        detections = json.load(f)

    result = predict_frame(image_path, clf, detections, {})
    out = Path(image_path).stem + "_predicted.png"
    cv2.imwrite(out, result)
    print(f"Saved → {out}")


def predict_all(output_dir: str):
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    with open(DETECTIONS) as f:
        detections = json.load(f)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    frames_dir = Path("frames")
    frame_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
    prev_dists: dict = {}

    for i, frame_path in enumerate(frame_files):
        result = predict_frame(str(frame_path), clf, detections, prev_dists)
        cv2.imwrite(str(out_path / frame_path.name), result)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1} / {len(frame_files)} frames done")

    print(f"Done. {len(frame_files)} frames saved to {output_dir}/")


# ── Main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--train",       action="store_true", help="Train the classifier")
    group.add_argument("--predict",     metavar="FRAME",     help="Predict on a single frame")
    group.add_argument("--predict-all", metavar="OUTPUT_DIR",help="Predict on all frames")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        predict_one(args.predict)
    elif args.predict_all:
        predict_all(args.predict_all)
