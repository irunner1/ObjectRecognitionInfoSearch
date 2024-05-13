"""Detection module."""

import cv2
from ultralytics import YOLO

from modules.search import search_info
from util.logger import configure_logger

logger = configure_logger(__name__)


class Detector:
    """Detect objects on video.

    Args:
        self.model: yolo model name
        self.video_path: Path to video or 0 for vebcam
    """

    def __init__(self, model_name: str, video_path: str | int) -> None:
        self.model = self.load_model(model_name=model_name)
        self.video_path = video_path

    def load_model(self, model_name: str) -> YOLO:
        """Set up model for object detection.
        If model not installed, it will be installed

        Args:
            model_name: name of yolo model, example: yolov5x6

        Returns:
            model: yolo model
        """

        model = YOLO("yolo/{name}.pt".format(name=model_name))
        model.fuse()
        model.conf = 0.5
        logger.info("model loaded")
        return model

    def on_video(self) -> None:
        """Launch object tracking and info search."""

        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            logger.error("Error opening file")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Error during recognition")
                break

            results = self.model(frame)
            class_names = set()
            for box in results[0].boxes:
                class_id = int(box.cls)
                class_label = results[0].names[class_id]

                logger.info(f"Detected class: {class_label}")
                class_names.add(class_label)

            annotated_frame = results[0].plot()

            cv2.imshow("YOLO Inference", annotated_frame)

            logger.info(f"Objects in frame: {class_names}")
            for cl in class_names:
                data = search_info(object_name=f"what is a {cl}")
                if "title" in data[0]:
                    logger.info(
                        "Founded data: {class_name} \n{title} \nlink: {link} \ndescription: {desc}".format(
                            class_name=cl,
                            title=data[0]["title"],
                            link=data[0]["link"],
                            desc=data[0]["snippet"],
                        )
                    )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("detection interrupted")
                break

        cap.release()
        cv2.destroyAllWindows()
