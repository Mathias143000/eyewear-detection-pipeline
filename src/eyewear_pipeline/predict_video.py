from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .config import ProjectConfig
from .inference import EyewearPredictor, draw_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict eyewear on video")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/prediction.mp4"))
    parser.add_argument("--model-type", choices=["baseline", "torch"], default="baseline")
    parser.add_argument("--model-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ProjectConfig()
    model_path = args.model_path or (
        cfg.baseline_model_path if args.model_type == "baseline" else cfg.torch_model_path
    )
    predictor = EyewearPredictor(model_path=model_path, model_type=args.model_type)

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        preds = predictor.predict_image(frame)
        writer.write(draw_predictions(frame, preds))

    cap.release()
    writer.release()
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

