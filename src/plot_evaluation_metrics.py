import json
import matplotlib.pyplot as plt
import os
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "../data/output")

face_models = ["haar", "dnn", "mtcnn", "yolo"]
plate_models = ["yolo_plate", "wpod", "rtdetr_plate"]

def load_results(models, base_folder):
    results = {}
    for model in models:
        path = os.path.join(base_folder, model, "results.json")
        if os.path.exists(path):
            with open(path) as f:
                results[model.upper()] = json.load(f)
        else:
            print(f"Warning: {path} not found!")
    return results

def extract_metrics(results):
    metrics = {}
    for model, data in results.items():
        m = data.get("_metrics", {})
        times = [v["time_sec"] for k, v in data.items() if k != "_metrics"]
        avg_time = np.mean(times) if times else 0
        metrics[model] = {
            "precision": m.get("precision", 0),
            "recall": m.get("recall", 0),
            "avg_time": avg_time
        }
    return metrics

def plot_metrics(metrics, title):
    models = list(metrics.keys())
    precision = [metrics[m]["precision"] for m in models]
    recall = [metrics[m]["recall"] for m in models]
    avg_time = [metrics[m]["avg_time"] for m in models]

    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    ax.bar(x, recall, width, label='Recall', color='lightgreen')
    ax.bar(x + width, avg_time, width, label='Avg Time (s)', color='salmon')

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# Load results
face_results = load_results(face_models, DATA)
plate_results = load_results(plate_models, DATA)

# Extract metrics
face_metrics = extract_metrics(face_results)
plate_metrics = extract_metrics(plate_results)

# Plot graphs
plot_metrics(face_metrics, "Face Detection Models Comparison")
plot_metrics(plate_metrics, "License Plate Detection Models Comparison")
