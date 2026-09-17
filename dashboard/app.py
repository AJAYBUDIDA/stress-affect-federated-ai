"""
AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI
Vercel-Compatible Flask Backend Web Application
Reads precomputed predictions and results using Python standard library (os, json, csv) + Flask.
"""

import os
import json
import csv
from flask import Flask, render_template, jsonify, send_from_directory, request

# Define base directory relative to this app.py file
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Initialize Flask App
app = Flask(__name__, template_folder="templates", static_folder="static")

# Data Cache
PREDICTIONS_CACHE = None


def load_precomputed_predictions():
    """
    Loads precomputed predictions from results/dashboard_predictions.json into memory.
    """
    global PREDICTIONS_CACHE
    json_path = os.path.join(BASE_DIR, "results", "dashboard_predictions.json")
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            PREDICTIONS_CACHE = json.load(f)
    else:
        PREDICTIONS_CACHE = None


# Load precomputed predictions on startup
load_precomputed_predictions()


# --- ROUTES ---

@app.route("/")
def index():
    """Renders main dashboard page."""
    return render_template("index.html")


@app.route("/results/plots/<filename>")
def serve_plot(filename):
    """Serves generated evaluation & FL plot images."""
    plots_dir = os.path.join(BASE_DIR, "results", "plots")
    return send_from_directory(plots_dir, filename)


@app.route("/api/summary")
def get_summary():
    """Returns overview metadata & dataset information."""
    return jsonify({
        "title": "AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI",
        "mode": "REAL-TIME SIMULATION — WESAD",
        "dataset": {
            "name": "WESAD (Wearable Stress and Affect Detection)",
            "subjects": 15,
            "subject_ids": ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17'],
            "total_windows": 6598,
            "extracted_features": 68,
            "class_counts": {
                "Neutral": 3512,
                "Stress": 1981,
                "Amusement": 1105
            }
        },
        "federated_architecture": {
            "clients": 3,
            "client_1": "S2, S3, S4, S5, S6 (2,161 samples)",
            "client_2": "S7, S8, S9, S10, S11 (2,211 samples)",
            "client_3": "S13, S14, S15 (1,331 samples)",
            "unseen_test_subjects": "S16, S17 (895 samples)",
            "total_test_windows": 895,
            "aggregation": "FedAvg (5 Rounds)",
            "raw_data_shared": False
        }
    })


@app.route("/api/simulation/windows")
def get_simulation_windows():
    """Returns list of available simulation windows from unseen test subjects S16 and S17."""
    global PREDICTIONS_CACHE
    if PREDICTIONS_CACHE is None:
        load_precomputed_predictions()

    if PREDICTIONS_CACHE is None or "predictions" not in PREDICTIONS_CACHE:
        return jsonify({"error": "Precomputed predictions file results/dashboard_predictions.json not found"}), 500

    predictions = PREDICTIONS_CACHE["predictions"]
    windows_list = [
        {
            "index": p["window_index"],
            "window_id": p["window_id"],
            "subject_id": p["subject_id"],
            "label_name": p["true_label"]
        }
        for p in predictions
    ]
    return jsonify({
        "total_test_windows": len(windows_list),
        "windows": windows_list
    })


@app.route("/api/simulation/predict/<int:window_idx>")
def predict_window(window_idx):
    """
    Returns precomputed PyTorch FedAvg model inference for a specific WESAD test window sample.
    Reads from results/dashboard_predictions.json without requiring PyTorch/heavy dependencies.
    """
    global PREDICTIONS_CACHE
    if PREDICTIONS_CACHE is None:
        load_precomputed_predictions()

    if PREDICTIONS_CACHE is None or "predictions" not in PREDICTIONS_CACHE:
        return jsonify({"error": "Precomputed predictions file results/dashboard_predictions.json not found"}), 500

    predictions = PREDICTIONS_CACHE["predictions"]
    if window_idx < 0 or window_idx >= len(predictions):
        window_idx = 0

    return jsonify(predictions[window_idx])


@app.route("/api/xai/importance")
def get_xai_importance():
    """Returns top SHAP feature importance rankings from results/xai/shap_feature_importance.csv."""
    csv_path = os.path.join(BASE_DIR, "results", "xai", "shap_feature_importance.csv")
    if not os.path.exists(csv_path):
        return jsonify({"error": "SHAP feature importance CSV not found"}), 404

    features_list = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            features_list.append({
                "feature": row["feature"],
                "mean_abs_shap": float(row["mean_abs_shap"])
            })

    return jsonify(features_list[:15])


@app.route("/api/xai/sample_explanation")
def get_xai_sample_explanation():
    """Returns contents of results/xai/sample_explanation.json."""
    json_path = os.path.join(BASE_DIR, "results", "xai", "sample_explanation.json")
    if not os.path.exists(json_path):
        return jsonify({"error": "Sample explanation JSON not found"}), 404

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return jsonify(data)


@app.route("/api/performance/comparison")
def get_performance_comparison():
    """Returns actual computed evaluation metrics comparing Centralized Baseline vs Federated Learning."""
    fl_json_path = os.path.join(BASE_DIR, "results", "federated_metrics.json")
    cent_json_path = os.path.join(BASE_DIR, "results", "centralized_metrics.json")

    if not os.path.exists(fl_json_path):
        return jsonify({"error": "Federated metrics file results/federated_metrics.json not found"}), 404

    if not os.path.exists(cent_json_path):
        return jsonify({"error": "Centralized metrics file results/centralized_metrics.json not found"}), 404

    with open(fl_json_path, 'r', encoding='utf-8') as f:
        fl_metrics = json.load(f)

    with open(cent_json_path, 'r', encoding='utf-8') as f:
        cent_metrics = json.load(f)

    cent_acc = cent_metrics.get("accuracy")
    fl_acc = fl_metrics.get("accuracy")

    def format_val(val, is_pct=False):
        if isinstance(val, (int, float)):
            if is_pct and val <= 1.0:
                return f"{val * 100:.2f}%"
            elif is_pct:
                return f"{val:.2f}%"
            return f"{val:.4f}"
        return str(val)

    return jsonify({
        "centralized": {
            "accuracy": format_val(cent_acc, is_pct=True),
            "precision_macro": format_val(cent_metrics.get("precision_macro")),
            "recall_macro": format_val(cent_metrics.get("recall_macro")),
            "f1_macro": format_val(cent_metrics.get("f1_macro")),
            "f1_weighted": format_val(cent_metrics.get("f1_weighted"))
        },
        "federated": {
            "accuracy": format_val(fl_acc, is_pct=True),
            "precision_macro": format_val(fl_metrics.get("precision_macro")),
            "recall_macro": format_val(fl_metrics.get("recall_macro")),
            "f1_macro": format_val(fl_metrics.get("f1_macro")),
            "f1_weighted": format_val(fl_metrics.get("f1_weighted"))
        }
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
