import cv2

from services.search import search_info
from util.logger import configure_logger

logger = configure_logger(__name__)


class Detector:
    """Detect objects on video.

    Args:
        self.video_path: Path to video. 0 - vebcam
        self.model: yolo model
    """

    def __init__(self, model, video_path) -> None:
        self.model = model
        self.video_path = video_path
        self.model.conf = 0.5

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

            for box in results[0].boxes:
                class_id = int(box.cls)
                class_label = results[0].names[class_id]
                logger.info(f"Detected class: {class_label}")
                # data = search_info((class_label))
                # logger.info(
                #     "Founded data: {title} \nlink: {link} \ndesctiption: {desc}".format(
                #         title=data[0]["title"],
                #         link=data[0]["link"],
                #         desc=data[0]["snippet"],
                #     )
                # )

            annotated_frame = results[0].plot()

            cv2.imshow("YOLO Inference", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("detection interrupted")
                break
        cap.release()
        cv2.destroyAllWindows()
