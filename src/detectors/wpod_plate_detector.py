import cv2
import numpy as np
from keras.models import model_from_json
import keras
from wpod_utils import detect_lp, DLabel

class WPODPlateDetector:
    def __init__(
        self,
        model_json="models/wpod-net.json",
        model_weights="models/wpod-net.h5",
        lp_threshold=0.3
    ):
        with open(model_json, "r") as f:
            model_json_str = f.read()

        self.model = model_from_json(
            model_json_str,
            custom_objects={"Model": keras.models.Model}
        )
        self.model.load_weights(model_weights)
        self.lp_threshold = lp_threshold

    def detect(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return []

        h, w = img.shape[:2]
        bound_dim = 1024

        try:
            labels, TLp, lp_type, Cor = detect_lp(
                self.model, img, bound_dim, lp_threshold=self.lp_threshold
            )
        except AssertionError:
            return []
        except Exception as e:
            print(f"WPODPlateDetector error: {e}")
            return []

        results = []

        for label in labels:
            if not isinstance(label, DLabel):
                continue
            tl = label.tl()
            br = label.br()
            if tl is None or br is None:
                continue
            x1, y1 = tl
            x2, y2 = br
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
            results.append((x1, y1, x2 - x1, y2 - y1))

        return results
