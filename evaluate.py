"""Evaluate trained Hangul detection weights on a separately reserved test split."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent / "outputs")
    args = parser.parse_args()
    if not args.model.is_file() or not args.data.is_file():
        parser.error("The trained model and dataset YAML must exist")
    if args.imgsz < 1:
        parser.error("imgsz must be positive")
    import yaml
    from ultralytics import YOLO

    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("test"):
        parser.error("Set a separate test split in the dataset YAML before evaluation")
    model = YOLO(str(args.model))
    if len(model.names) != 1:
        parser.error("Expected locally trained single-class Hangul text-region weights")
    options = dict(data=str(args.data.resolve()), split="test", imgsz=args.imgsz,
                   project=str(args.project.resolve()), name="test")
    if args.device is not None:
        options["device"] = args.device
    metrics = model.val(**options)
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
