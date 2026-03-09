"""
Standalone script to run YOLOv8 clothing detection on an image file.

Usage:
    python scripts/run_yolo_detection.py --image_path path/to/image.jpg
"""
import argparse
from pathlib import Path

from PIL import Image

from app.services.detection import DetectionService
from ml.yolo_detector import get_yolo_detector


def run_detection(image_path: str):
    """Run YOLOv8 detection on the given image file."""
    # Load the image
    image = Image.open(image_path)

    # Initialize the detection service
    detector = get_yolo_detector()
    service = DetectionService(detector)

    # Run detection
    detections = service.detect_clothing(image)

    # Print results
    print("Detections:")
    print(detections)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 clothing detection on an image file.")
    parser.add_argument(
        "--image_path", type=str, required=True, help="Path to the image file to process."
    )
    args = parser.parse_args()

    # Ensure the image file exists
    image_path = Path(args.image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Run detection
    run_detection(str(image_path))