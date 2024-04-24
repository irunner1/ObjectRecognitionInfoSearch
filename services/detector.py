import time

import cv2
import numpy as np

from services.search import search_info
from util.logger import configure_logger

logger = configure_logger(__name__)
np.random.seed(10)


class Detector:
    """Detect objects on video.

    Args:
        self.video_path = Путь к видео
        self.config_path = Путь к настройкам
        self.model_path = Путь к модели
        self.classes_path = Путь к классам для распознания
    """

    def __init__(self, video_path, config_path, model_path, classes_path) -> None:
        self.classes_list = None
        self.color_list = None
        self.video_path = video_path
        self.config_path = config_path
        self.model_path = model_path
        self.classes_path = classes_path
        self.net = cv2.dnn_DetectionModel(self.model_path, self.config_path)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.read_class()

    def read_class(self):

        with open(self.classes_path, "r") as f:
            self.classes_list = f.read().splitlines()
        self.classes_list.insert(0, "__Background__")
        self.color_list = np.random.uniform(
            low=0, high=255, size=(len(self.classes_list), 3)
        )

    def on_video(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            logger.error("Error opening file")
            return

        (success, image) = cap.read()
        while success:
            class_label_ids, confidences, bboxs = self.net.detect(
                image, confThreshold=0.5
            )
            bboxs = list(bboxs)
            confidences = list(
                map(float, list(np.array(confidences).reshape(1, -1)[0]))
            )

            bbox_idxs = cv2.dnn.NMSBoxes(
                bboxs, confidences, score_threshold=0.5, nms_threshold=0.2
            )
            if len(bbox_idxs) != 0:
                for i in range(0, len(bbox_idxs)):
                    bbox = bboxs[np.squeeze(bbox_idxs[i])]
                    class_confidence = confidences[np.squeeze(bbox_idxs[i])]
                    class_label_id = np.squeeze(
                        class_label_ids[np.squeeze(bbox_idxs[i])]
                    )
                    class_label = self.classes_list[class_label_id]
                    class_color = [
                        int(color) for color in self.color_list[class_label_id]
                    ]
                    displayText = "{label}: {conf:.2f}".format(
                        label=class_label, conf=class_confidence
                    )
                    logger.info("class {text}".format(text=displayText))
                    data = search_info((class_label))
                    logger.info(
                        "Founded data: {title} \nlink: {link} \ndesctiption: {desc}".format(
                            title=data[0]["title"],
                            link=data[0]["link"],
                            desc=data[0]["snippet"],
                        )
                    )

                    x, y, w, h = bbox
                    cv2.rectangle(
                        image,
                        (x, y),
                        (x + w, y + h),
                        color=class_color,
                        thickness=2,
                    )
                    # cv2.putText(
                    #     image,
                    #     displayText,
                    #     (x, y - 10),
                    #     cv2.FONT_HERSHEY_PLAIN,
                    #     1,
                    #     class_color,
                    #     2,
                    # )
            cv2.imshow("Video object recognition and info search", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()
        cv2.destroyAllWindows()
