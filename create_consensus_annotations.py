#!/usr/bin/env python3
"""
Create a consensus annotations.json from multiple annotators using majority voting.

For each frame and each pair of people, each annotator votes same-group (1) or
different-group (0). If the majority vote is same-group, the pair is treated as
same-group in the consensus. Groups are then reconstructed via connected components.

Usage:
    python create_consensus_annotations.py
    python create_consensus_annotations.py --output frames/consensus_annotations.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


ANNOTATOR_FILES = [
    "frames/annotations.json",
    "frames/groupAnnotations/hieuAnnotations.json",
    "frames/groupAnnotations/jeslynAnnotations.json",
]


def pairwise_votes(frame_data: list[dict]) -> dict[tuple, int]:
    """Return {(pid_i, pid_j): 1 if same group else 0} for all pairs in a frame."""
    labeled = {p["person_id"]: p["group_id"] for p in frame_data if p["group_id"] is not None}
    pids = sorted(labeled.keys())
    votes = {}
    for a in range(len(pids)):
        for b in range(a + 1, len(pids)):
            i, j = pids[a], pids[b]
            key = (i, j)
            votes[key] = int(labeled[i] == labeled[j])
    return votes


def connected_components(pids: list, same_group_pairs: list[tuple]) -> dict:
    """Assign group IDs via union-find. Returns {person_id: group_id or None}."""
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

    roots = {}
    gid = 1
    result = {}
    for p in pids:
        r = find(p)
        if r not in roots:
            roots[r] = gid
            gid += 1
        result[p] = roots[r]

    grouped_pids = {p for pair in same_group_pairs for p in pair}
    for p in pids:
        if p not in grouped_pids:
            result[p] = None

    return result


def majority_vote(vote_lists: list[dict[tuple, int]], threshold: float = 0.5) -> list[tuple]:
    """Return pairs where the fraction of same-group votes exceeds threshold."""
    pair_votes = defaultdict(list)
    for votes in vote_lists:
        for pair, vote in votes.items():
            pair_votes[pair].append(vote)

    same_group = []
    for pair, votes in pair_votes.items():
        if sum(votes) / len(votes) > threshold:
            same_group.append(pair)
    return same_group


def build_consensus(annotator_files: list[str], output_path: str, threshold: float = 0.5):
    # Load all annotation files
    all_data = []
    for f in annotator_files:
        with open(f) as fp:
            all_data.append(json.load(fp))
        print(f"Loaded {f}  ({len(all_data[-1])} frames)")

    # Use frames present in all annotators
    frame_sets = [set(d.keys()) for d in all_data]
    common_frames = sorted(set.intersection(*frame_sets))
    print(f"\nFrames annotated by all {len(annotator_files)} annotators: {len(common_frames)}")

    consensus = {}
    pair_agreement_scores = []

    for frame_name in common_frames:
        # Collect all person IDs and bboxes from first annotator (bboxes are the same across all)
        persons_ref = all_data[0][frame_name]
        all_pids = [p["person_id"] for p in persons_ref]
        bbox_map = {p["person_id"]: p["bbox"] for p in persons_ref}

        # Each annotator votes on pairs
        vote_lists = []
        for data in all_data:
            frame_data = data.get(frame_name, [])
            vote_lists.append(pairwise_votes(frame_data))

        # Compute agreement per pair for stats
        all_pairs = set()
        for v in vote_lists:
            all_pairs.update(v.keys())
        for pair in all_pairs:
            votes = [v[pair] for v in vote_lists if pair in v]
            if len(votes) > 1:
                frac = sum(votes) / len(votes)
                pair_agreement_scores.append(abs(frac - 0.5) * 2)  # 1=full agree, 0=split

        # Majority vote → same-group pairs
        same_group_pairs = majority_vote(vote_lists, threshold=threshold)

        # Reconstruct group IDs
        group_map = connected_components(all_pids, same_group_pairs)

        consensus[frame_name] = [
            {
                "person_id": pid,
                "bbox": bbox_map[pid],
                "group_id": group_map.get(pid),
            }
            for pid in all_pids
        ]

    with open(output_path, "w") as f:
        json.dump(consensus, f, indent=2)

    avg_agreement = sum(pair_agreement_scores) / len(pair_agreement_scores) if pair_agreement_scores else 0
    print(f"Average pairwise inter-annotator agreement: {avg_agreement:.1%}")
    print(f"Consensus annotations saved to {output_path}")
    print(f"  {len(consensus)} frames ready for training")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="frames/consensus_annotations.json")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Fraction of same-group votes needed (default 0.5 = majority)")
    args = parser.parse_args()

    build_consensus(ANNOTATOR_FILES, args.output, args.threshold)
