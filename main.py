import os

from services.detector import Detector


def main():
    video_path = "test_videos/streets_nyc.mp4"
    # video_path = 0  # vebcam
    config_path = os.path.join(
        "model_data",
        "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt",
        # "model_data",
        # "yolov5x6.pt",
    )
    model_path = os.path.join("model_data", "frozen_inference_graph.pb")
    classes_path = os.path.join("model_data", "coco.names")

    detector = Detector(
        video_path=video_path,
        config_path=config_path,
        model_path=model_path,
        classes_path=classes_path,
    )
    detector.onVideo()


if __name__ == "__main__":
    main()
