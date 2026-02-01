import cv2
import numpy as np

class DNNFaceDetector:
    def __init__(self, model_path, proto_path, conf=0.15):
        self.net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        self.conf = conf

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False
        )

        self.net.setInput(blob)
        detections = self.net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.conf:
                box = detections[0, 0, i, 3:7] * [w, h, w, h]
                x1, y1, x2, y2 = box.astype(int)
                faces.append((x1, y1, x2 - x1, y2 - y1))

        return faces
