import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, AutoModelForObjectDetection
from PIL import Image


class RTDETRPlateDetector:
    def __init__(
        self,
        model_name="Garon16/rtdetr_r50vd_russia_plate_detector",
        conf=0.6,
        device="cpu"
    ):
        self.device = torch.device(device)

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.conf = conf

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []

        h, w = img.shape[:2]
        image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_object_detection(
            outputs,
            threshold=self.conf,
            target_sizes=[(h, w)]
        )[0]

        plates = []
        min_area = (w * h) * 0.002

        for box, score in zip(results["boxes"], results["scores"]):
            x1, y1, x2, y2 = box.cpu().numpy()
            w_box = x2 - x1
            h_box = y2 - y1

            if w_box <= 0 or h_box <= 0:
                continue

            area = w_box * h_box
            if area < min_area:
                continue

            aspect_ratio = w_box / h_box
            if aspect_ratio < 1.8 or aspect_ratio > 6.5:
                continue

            plates.append((
                int(x1),
                int(y1),
                int(w_box),
                int(h_box)
            ))

        return plates

