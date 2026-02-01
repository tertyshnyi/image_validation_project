import os, json
from detectors.haar_detector import HaarFaceDetector
from detectors.dnn_detector import DNNFaceDetector
from detectors.mtcnn_detector import MTCNNFaceDetector
from detectors.yolo_detector import YOLOFaceDetector
from evaluate import evaluate
from evaluate_plate import evaluate_plate
from detectors.yolo_plate_detector import YOLOPlateDetector
from detectors.wpod_plate_detector import WPODPlateDetector
from detectors.rtdetr_plate_detector import RTDETRPlateDetector

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "../data")
MODELS = os.path.join(BASE, "../models")

with open(os.path.join(DATA, "ground_truth_face.json")) as f:
    gt_faces = json.load(f)

with open(os.path.join(DATA, "ground_truth_plate.json")) as f:
    gt_plates = json.load(f)

evaluate_plate(
    YOLOPlateDetector(os.path.join(MODELS, "yolo_plate.pt")),
    "YOLO_PLATE",
    os.path.join(DATA, "input"),
    os.path.join(DATA, "output/yolo_plate"),
    gt_plates
)

evaluate_plate(
    WPODPlateDetector(
        model_json=os.path.join(MODELS, "wpod-net.json"),
        model_weights=os.path.join(MODELS, "wpod-net.h5")
    ),
    "WPOD_NET",
    os.path.join(DATA, "input"),
    os.path.join(DATA, "output/wpod"),
    gt_plates
)

evaluate_plate(
    RTDETRPlateDetector(conf=0.6),
    "RT_DETR_PLATE",
    os.path.join(DATA, "input"),
    os.path.join(DATA, "output/rtdetr_plate"),
    gt_plates
)

evaluate(
    HaarFaceDetector(os.path.join(MODELS, "haarcascade_frontalface_default.xml")),
    "HAAR",
    os.path.join(DATA, "input"),
    os.path.join(DATA, "output/haar"),
    gt_faces
)

evaluate(
    DNNFaceDetector(
        os.path.join(MODELS, "res10_300x300_ssd_iter_140000.caffemodel"),
        os.path.join(MODELS, "deploy.prototxt")
    ),
    "DNN",
    os.path.join(DATA, "input"),
    os.path.join(DATA, "output/dnn"),
    gt_faces
)

evaluate(
    MTCNNFaceDetector(),
    "MTCNN",
    os.path.join(DATA, "input"),
    os.path.join(DATA, "output/mtcnn"),
    gt_faces
)

evaluate(
    YOLOFaceDetector(os.path.join(MODELS, "yolov8n.pt")),
    "YOLO",
    os.path.join(DATA, "input"),
    os.path.join(DATA, "output/yolo"),
    gt_faces
)
