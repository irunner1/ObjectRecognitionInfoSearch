import time

import cv2
import numpy as np

np.random.seed(10)


class Detector:
    """Detect objects on video.

    Args:
        self.video_path = Путь до файла с видео
        self.config_path = Путь до настроек
        self.model_path = Путь до модели
        self.classes_path = Путь до классов
    """

    def __init__(self, video_path, config_path, model_path, classes_path) -> None:
        self.video_path = video_path
        self.config_path = config_path
        self.model_path = model_path
        self.classes_path = classes_path
        self.net = cv2.dnn_DetectionModel(self.model_path, self.config_path)
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)

        self.readClass()

    def readClass(self):

        with open(self.classes_path, "r") as f:
            self.classes_list = f.read().splitlines()
        self.classes_list.insert(0, "__Background__")
        self.color_list = np.random.uniform(
            low=0, high=255, size=(len(self.classes_list), 3)
        )
        # print(self.classes_list)

    def onVideo(self):
        cap = cv2.VideoCapture(self.video_path)

        if cap.isOpened() == False:
            print("Error opening file")
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

                    x, y, w, h = bbox
                    cv2.rectangle(
                        image,
                        (x, y),
                        (x + w, y + h),
                        color=class_color,
                        thickness=2,
                    )
                    cv2.putText(
                        image,
                        displayText,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_PLAIN,
                        1,
                        class_color,
                        2,
                    )
            cv2.imshow("Result", image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            (success, image) = cap.read()
        cv2.destroyAllWindows()
