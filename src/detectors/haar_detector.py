import cv2

class HaarFaceDetector:
    def __init__(self, model_path):
        self.detector = cv2.CascadeClassifier(model_path)
        if self.detector.empty():
            raise RuntimeError("Cannot load Haar model")

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(40, 40)
        )

        return [(x, y, w, h) for (x, y, w, h) in faces]
