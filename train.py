"""Train YOLOv8n to detect Hangul handwriting regions (not OCR)."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Your dataset's data.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained name or local .pt file")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="Examples: 0, cpu; default auto-detects")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--name", default="train")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset YAML not found: {args.data}")
    if min(args.epochs, args.imgsz, args.batch) < 1 or args.workers < 0:
        parser.error("epochs, imgsz, batch must be positive; workers must be nonnegative")
    # Avoid downloading model weights until an actual dataset file was supplied.
    import yaml
    from ultralytics import YOLO

    config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not config.get("train") or not config.get("val"):
        parser.error("Dataset YAML must define train, val and one Hangul region class")
    names = config.get("names", [])
    if not isinstance(names, (list, dict)) or len(names) != 1:
        parser.error("This project expects exactly one detection class (Hangul text region)")
    model = YOLO(args.model)
    options = dict(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                   batch=args.batch, seed=args.seed, deterministic=True, workers=args.workers,
                   project=str(args.project.resolve()), name=args.name)
    if args.device is not None:
        options["device"] = args.device
    model.train(**options)
    print(f"Run directory: {model.trainer.save_dir}")
    print(f"Best model: {model.trainer.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
