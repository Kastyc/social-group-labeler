#!/usr/bin/env python3
"""
Extract image frames from a ROS2 bag file at a target FPS.

Usage:
    python extract_frames.py path/to/recording.bag ./output_frames
    python extract_frames.py path/to/recording.bag ./output_frames --fps 5 --topic /camera/image_raw

Requires: pip install rosbags opencv-python numpy
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def extract_frames(bag_path: str, output_dir: str, fps: float = 5.0, topic: str | None = None):
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError:
        sys.exit("rosbags not installed. Run: pip install rosbags")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    frame_interval_ns = int((1.0 / fps) * 1e9)

    with Reader(bag_path) as reader:
        # Discover image topics
        image_connections = [
            c for c in reader.connections
            if ("Image" in c.msgtype or "CompressedImage" in c.msgtype)
            and (topic is None or c.topic == topic)
        ]

        if not image_connections:
            all_topics = [(c.topic, c.msgtype) for c in reader.connections]
            print("No image topics found. Available topics:")
            for t, m in all_topics:
                print(f"  {t}  [{m}]")
            sys.exit(1)

        print("Extracting from topics:")
        for c in image_connections:
            print(f"  {c.topic}  [{c.msgtype}]")

        last_saved_ns: dict[str, int] = {}
        frame_count: dict[str, int] = {}

        for connection, timestamp, rawdata in reader.messages(connections=image_connections):
            t = connection.topic

            if t in last_saved_ns and (timestamp - last_saved_ns[t]) < frame_interval_ns:
                continue

            last_saved_ns[t] = timestamp
            frame_count[t] = frame_count.get(t, 0) + 1

            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            img = _decode_image(msg, connection.msgtype)
            if img is None:
                continue

            # Build a safe filename from the topic
            safe_topic = t.replace("/", "_").strip("_")
            fname = f"{safe_topic}_{frame_count[t]:06d}.png"
            cv2.imwrite(str(output_path / fname), img)

            if frame_count[t] % 50 == 0:
                print(f"  {t}: {frame_count[t]} frames extracted")

    total = sum(frame_count.values())
    print(f"\nDone. {total} frames written to: {output_dir}")


def _decode_image(msg, msgtype: str):
    """Convert a ROS image message to a BGR numpy array."""
    if "CompressedImage" in msgtype:
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    # Raw Image
    encoding = getattr(msg, "encoding", "").lower()
    h, w = msg.height, msg.width
    data = np.frombuffer(msg.data, dtype=np.uint8)

    try:
        if encoding in ("rgb8", "rgb"):
            return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        elif encoding in ("bgr8", "bgr"):
            return data.reshape(h, w, 3)
        elif encoding in ("rgba8",):
            return cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
        elif encoding in ("bgra8",):
            return cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_BGRA2BGR)
        elif encoding in ("mono8", "8uc1"):
            return cv2.cvtColor(data.reshape(h, w), cv2.COLOR_GRAY2BGR)
        elif encoding in ("mono16", "16uc1"):
            data16 = np.frombuffer(msg.data, dtype=np.uint16).reshape(h, w)
            norm = (data16 / 256).astype(np.uint8)
            return cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        elif encoding in ("bayer_rggb8", "bayer_bggr8", "bayer_gbrg8", "bayer_grbg8"):
            codes = {
                "bayer_rggb8": cv2.COLOR_BayerBG2BGR,
                "bayer_bggr8": cv2.COLOR_BayerRG2BGR,
                "bayer_gbrg8": cv2.COLOR_BayerGR2BGR,
                "bayer_grbg8": cv2.COLOR_BayerGB2BGR,
            }
            return cv2.cvtColor(data.reshape(h, w), codes[encoding])
        else:
            print(f"  [warn] Unsupported encoding '{encoding}', skipping frame.")
            return None
    except Exception as e:
        print(f"  [warn] Could not decode frame ({encoding}): {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract PNG frames from a ROS2 bag file.")
    parser.add_argument("bag_path", help="Path to the .bag file or bag directory")
    parser.add_argument("output_dir", help="Directory to write PNG frames into")
    parser.add_argument("--fps", type=float, default=5.0, help="Target frames per second (default: 5)")
    parser.add_argument("--topic", default=None, help="Specific image topic to extract (default: all image topics)")
    args = parser.parse_args()

    extract_frames(args.bag_path, args.output_dir, fps=args.fps, topic=args.topic)
