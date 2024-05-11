"""Run project with object detection and searhing info."""

from ultralytics import YOLO

from services.detector import Detector
from util.logger import configure_logger

logger = configure_logger(__name__)


def main() -> None:
    """Launch objects detection on video."""

    video_path = "test_videos/streets_nyc.mp4"  # 0 - vebcam
    logger.info("video prepared")
    model = YOLO("yolo/yolov5x6u.pt")
    logger.info("model loaded")

    detector = Detector(model=model, video_path=video_path)
    detector.on_video()


if __name__ == "__main__":
    main()
