from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .config import ProjectConfig
from .inference import EyewearPredictor, draw_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict eyewear on image")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/prediction.jpg"))
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

    image = cv2.imread(str(args.image))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")
    preds = predictor.predict_image(image)
    vis = draw_predictions(image, preds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), vis)

    payload = [
        {
            "label_id": p.label_id,
            "label_name": p.label_name,
            "confidence": p.confidence,
            "score_positive": p.score_positive,
            "bbox_xyxy": list(p.bbox_xyxy),
        }
        for p in preds
    ]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
