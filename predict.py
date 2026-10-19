"""Detect handwriting boxes in images using locally trained YOLO weights."""
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Trained weights/best.pt")
    parser.add_argument("--source", type=Path, required=True, help="Local image or image folder")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--name", default="predict")
    args = parser.parse_args()
    if not args.model.is_file() or not args.source.exists():
        parser.error("The trained model and the local source must exist")
    if not 0 < args.confidence <= 1 or args.imgsz < 1:
        parser.error("confidence must be in (0, 1]; imgsz must be positive")
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    if len(model.names) != 1:
        parser.error("Expected locally trained single-class Hangul text-region weights")
    options = dict(source=str(args.source.resolve()), conf=args.confidence, imgsz=args.imgsz,
                   save=True, save_txt=True, save_conf=True, stream=True,
                   project=str(args.project.resolve()), name=args.name)
    if args.device is not None:
        options["device"] = args.device
    for result in model.predict(**options):
        print(f"{result.path}: {len(result.boxes)} detected text regions; saved in {result.save_dir}")


if __name__ == "__main__":
    main()
