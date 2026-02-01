from ultralytics import YOLO
import cv2

class YOLOFaceDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, image_path):
        results = self.model.predict(source=image_path, verbose=False)
        faces = []

        img = cv2.imread(image_path)
        h, w = img.shape[:2]

        for r in results:
            for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
                label = r.names[int(cls)]
                if label == "person":
                    x1, y1, x2, y2 = map(int, box)
                    faces.append((x1, y1, x2 - x1, y2 - y1))

        return faces
