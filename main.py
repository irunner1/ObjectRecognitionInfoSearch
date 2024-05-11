"""Run project with object detection and searhing info."""

from base.settings import SETTINGS
from services.detector import Detector
from util.logger import configure_logger

logger = configure_logger(__name__)


def main() -> None:
    """Launch objects detection on video."""

    video_path = "test_videos/streets_nyc.mp4"  # 0 - vebcam
    logger.info("video prepared")

    detector = Detector(model_name=SETTINGS.model, video_path=video_path)
    detector.on_video()


if __name__ == "__main__":
    main()
