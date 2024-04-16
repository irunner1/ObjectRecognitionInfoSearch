import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk

from util.logger import configure_logger

logger = configure_logger(__name__)


def recognize_objects(frame):

    model = torch.hub.load("ultralytics/yolov5", "yolov5x6")
    if torch.cuda.is_available():
        logger.info("cuda available")
        model = model.cuda()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(image, (640, 640))
    img_resized = img_resized.transpose((2, 0, 1))
    img_resized = img_resized / 255.0
    img_resized = torch.from_numpy(img_resized).unsqueeze(0).float()

    if torch.cuda.is_available():
        img_resized = img_resized.cuda()

    results = model(img_resized)

    results = results[0].to("cpu").detach().numpy()
    for result in results:
        logger.info(result)
        x1, y1, x2, y2 = map(int, result[:4])
        cv2.rectangle(frame, (x1, y1), (x1 + x2, y1 + y2), (0, 255, 0), 2)
    return frame
    pass


def select_object(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONUP:
        pass


window = tk.Tk()
window.title("Video object recognition and info search")

canvas = tk.Canvas(window, width=1920, height=1080)
canvas.pack()

cap = cv2.VideoCapture("streets_nyc.mp4")
logger.info(cap)
while True:
    ret, frame = cap.read()
    logger.info(ret)
    logger.info(frame)
    if not ret:
        break
    cv2.setMouseCallback("frame", select_object)
    objects = recognize_objects(frame)
    cv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv_img)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas.imgtk = imgtk
    canvas.create_image(0, 0, anchor="nw", image=imgtk)
    window.update()

button = tk.Button(window, text="Select Object", command=select_object)
button.pack(side="bottom")

window.mainloop()
