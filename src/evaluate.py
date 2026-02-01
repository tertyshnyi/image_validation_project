import os, json, time
from sklearn.metrics import precision_score, recall_score
from draw_faces import draw_faces

def evaluate(detector, name, input_dir, output_dir, ground_truth):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    results = {}
    y_true, y_pred = [], []

    for img in os.listdir(input_dir):
        if not img.lower().endswith((".jpg",".png",".jpeg")):
            continue

        path = os.path.join(input_dir, img)
        start = time.time()
        faces = detector.detect(path)
        elapsed = time.time() - start

        detected = len(faces) > 0
        gt = ground_truth.get(img, {"faces": False})["faces"]

        y_true.append(gt)
        y_pred.append(detected)

        out_img = os.path.join(images_dir, img)
        draw_faces(path, faces, out_img)

        results[img] = {
            "faces_detected": detected,
            "faces_count": len(faces),
            "time_sec": round(elapsed, 3)
        }

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    results["_metrics"] = {
        "precision": round(precision, 3),
        "recall": round(recall, 3)
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    print(f"{name}: Precision={precision:.2f}, Recall={recall:.2f}")
