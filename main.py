import os

from services.detector import Detector
from util.logger import configure_logger

logger = configure_logger(__name__)


def main():
    video_path = "unused/test_videos/streets_nyc.mp4"  # 0 - vebcam
    config_path = os.path.join(
        "model_data",
        "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt",
    )
    model_path = os.path.join("model_data", "frozen_inference_graph.pb")
    classes_path = os.path.join("model_data", "coco.names")

    logger.info("settings loaded")
    detector = Detector(
        video_path=video_path,
        config_path=config_path,
        model_path=model_path,
        classes_path=classes_path,
    )
    detector.on_video()


if __name__ == "__main__":
    main()
