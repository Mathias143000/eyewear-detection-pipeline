from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create synthetic glasses/no_glasses dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--samples-per-class", type=int, default=120)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def draw_face(img: np.ndarray, rng: np.random.Generator, with_glasses: bool) -> np.ndarray:
    h, w = img.shape[:2]
    center = (w // 2 + int(rng.integers(-6, 7)), h // 2 + int(rng.integers(-6, 7)))
    radius = int(min(w, h) * 0.35)
    skin = (180 + int(rng.integers(-20, 21)), 190, 210)
    cv2.circle(img, center, radius, skin, -1)
    left_eye = (center[0] - radius // 3, center[1] - radius // 5)
    right_eye = (center[0] + radius // 3, center[1] - radius // 5)
    cv2.circle(img, left_eye, 5, (40, 40, 40), -1)
    cv2.circle(img, right_eye, 5, (40, 40, 40), -1)
    cv2.ellipse(img, (center[0], center[1] + radius // 4), (18, 10), 0, 0, 180, (55, 55, 55), 2)
    if with_glasses:
        lens_color = (15, 15, 15)
        cv2.circle(img, left_eye, 16, lens_color, 2)
        cv2.circle(img, right_eye, 16, lens_color, 2)
        cv2.line(img, (left_eye[0] + 16, left_eye[1]), (right_eye[0] - 16, right_eye[1]), lens_color, 2)
    return img


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    for label in ("glasses", "no_glasses"):
        (args.output_dir / label).mkdir(parents=True, exist_ok=True)

    for i in range(args.samples_per_class):
        for label, with_glasses in (("glasses", True), ("no_glasses", False)):
            img = np.zeros((args.size, args.size, 3), dtype=np.uint8)
            bg = int(rng.integers(110, 180))
            img[:] = (bg, bg, bg)
            face = draw_face(img, rng, with_glasses=with_glasses)
            noise = rng.normal(0, 9, size=face.shape).astype(np.int16)
            face = np.clip(face.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            out = args.output_dir / label / f"{label}_{i:04d}.png"
            cv2.imwrite(str(out), face)

    print(f"Synthetic dataset created at {args.output_dir}")


if __name__ == "__main__":
    main()

