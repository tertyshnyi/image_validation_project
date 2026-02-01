# Car Plate & Person Validator

Detects the presence of people and vehicle license plates in images using YOLOv8 and EasyOCR.  
The project evaluates multiple detection models and generates precision/recall metrics along with average processing times.

---

## Features
- Person detection
- License plate detection
- Draw bounding boxes on images
- Generate precision and recall metrics
- Compare multiple models using evaluation graphs

---

## Project Structure

image_validation_project/
├── src/
│ ├── run_all.py # Runs all detection models and generates JSON results
│ ├── plot_evaluation_metrics.py # Plots bar charts for precision, recall, and average time
│ └── detectors/ # Detection model implementations
├── data/
│ ├── input/ # Input images
│ ├── output/ # Results will be saved here
│ ├── ground_truth_face.json
│ └── ground_truth_plate.json
├── models/ # Pretrained models
├── requirements.txt
└── README.md


---

## Setup

Clone the repository and create a Python virtual environment:

```bash
git clone <repo>
cd image_validation_project
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Usage
1. Run the full evaluation pipeline
python src/run_all.py
This will:

Run all face/person and license plate detection models

Save results as JSON files in the output/ folders for each model, e.g.:

output/dnn/result.json
output/yolo/result.json
output/haar/result.json
output/yolo_plate/result.json
output/wpod/result.json
output/rtdetr_plate/result.json
2. Generate evaluation graphs
python src/plot_evaluation_metrics.py
This will:

Load the JSON results from the output/ folders

Plot bar charts comparing:

Precision

Recall

Average processing time (seconds)

One chart for face/person detection models, and one chart for license plate detection models

💡 Tip: If you want, you can modify the script to save the charts as PNG files using plt.savefig("filename.png").

Example Output
After running the evaluation and plotting scripts, you will see bar charts like:

Face Detection Models Comparison

HAAR, DNN, MTCNN, YOLO

Shows precision, recall, and average processing time

License Plate Detection Models Comparison

YOLO_PLATE, WPOD_NET, RT_DETR_PLATE

Shows precision, recall, and average processing time

Notes
Make sure all input images are in data/input/

Make sure the ground truth JSON files exist (ground_truth_face.json and ground_truth_plate.json)

All models should be downloaded or placed in the models/ folder