from ultralytics import YOLO
import cv2

class YOLOPlateDetector:
    def __init__(self, model_path="models/license-plate.pt", conf=0.3):
        """
        model_path – YOLOv8 model trained on license plates
        conf – confidence threshold
        """
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []

        results = self.model(
            img,
            conf=self.conf,
            verbose=False
        )

        plates = []

        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                w = x2 - x1
                h = y2 - y1
                plates.append((x1, y1, w, h))

        return plates
