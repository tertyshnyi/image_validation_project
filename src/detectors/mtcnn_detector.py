from mtcnn import MTCNN
import cv2

class MTCNNFaceDetector:
    def __init__(self, min_face_size=20):
        self.detector = MTCNN()
        self.min_face_size = min_face_size

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []

        try:
            results = self.detector.detect_faces(img)
        except Exception:
            return []

        faces = []
        for r in results:
            x, y, w, h = r['box']
            if w >= self.min_face_size and h >= self.min_face_size:
                faces.append((x, y, w, h))

        return faces

