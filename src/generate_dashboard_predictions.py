"""
AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI
Precomputed Dashboard Predictions Generation Module
Generates results/dashboard_predictions.json using actual PyTorch FedAvg model inferences.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import torch

# Add root directory to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import StressClassifier

CLASS_MAP_REV = {0: 'Neutral', 1: 'Stress', 2: 'Amusement'}


def generate_dashboard_predictions():
    """
    Loads WESAD dataset, filters for unseen test subjects S16 and S17,
    scales features using federated scaler, runs inferences via PyTorch FedAvg global model,
    and exports formatted JSON structure to results/dashboard_predictions.json.
    """
    csv_path = os.path.join("data", "processed", "wesad_features.csv")
    scaler_path = os.path.join("models", "federated_scaler.pkl")
    model_path = os.path.join("models", "federated_global_model.pth")
    output_json_path = os.path.join("results", "dashboard_predictions.json")

    # Check required source files
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"WESAD dataset file not found at {csv_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Federated scaler file not found at {scaler_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Federated global model file not found at {model_path}")

    # Load dataset & filter for test subjects S16 and S17
    df = pd.read_csv(csv_path)
    metadata_cols = ['subject_id', 'label', 'label_name', 'window_id']
    feature_cols = [c for c in df.columns if c not in metadata_cols]

    test_subjects = ['S16', 'S17']
    test_df = df[df['subject_id'].isin(test_subjects)].copy().reset_index(drop=True)

    # Load scaler and model
    scaler = joblib.load(scaler_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StressClassifier(input_dim=len(feature_cols), num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions_list = []

    print(f"Generating predictions for {len(test_df)} test windows across subjects {test_subjects}...")

    for idx, row in test_df.iterrows():
        # Extract raw features and scale
        x_raw = row[feature_cols].values.reshape(1, -1)
        x_scaled = scaler.transform(x_raw)

        # PyTorch Inference
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = model(x_tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class_idx = int(np.argmax(probs))

        pred_label_name = CLASS_MAP_REV[pred_class_idx]
        pred_confidence = float(probs[pred_class_idx])

        # Extract actual sensor metrics from dataset row
        sensor_display = {
            "wrist_temp_mean": round(float(row.get('w_temp_mean', 0.0)), 2),
            "wrist_temp_min": round(float(row.get('w_temp_min', 0.0)), 2),
            "wrist_temp_max": round(float(row.get('w_temp_max', 0.0)), 2),
            "wrist_hr_mean": round(float(row.get('w_hr_mean', 0.0)), 1),
            "wrist_hr_min": round(float(row.get('w_hr_min', 0.0)), 1),
            "wrist_hr_max": round(float(row.get('w_hr_max', 0.0)), 1),
            "wrist_eda_mean": round(float(row.get('w_eda_mean', 0.0)), 4),
            "chest_ecg_mean": round(float(row.get('c_ecg_mean', 0.0)), 4),
            "chest_ecg_std": round(float(row.get('c_ecg_std', 0.0)), 4),
            "chest_eda_mean": round(float(row.get('c_eda_mean', 0.0)), 4),
            "chest_resp_mean": round(float(row.get('c_resp_mean', 0.0)), 3),
            "chest_acc_mag_mean": round(float(row.get('c_acc_mag_mean', 0.0)), 3),
            "wrist_acc_mag_mean": round(float(row.get('w_acc_mag_mean', 0.0)), 3)
        }

        entry = {
            "window_index": idx,
            "window_id": str(row['window_id']),
            "subject_id": str(row['subject_id']),
            "true_label": str(row['label_name']),
            "predicted_label": pred_label_name,
            "prediction_confidence": round(pred_confidence * 100, 2),
            "class_probabilities": {
                "Neutral": round(float(probs[0]) * 100, 2),
                "Stress": round(float(probs[1]) * 100, 2),
                "Amusement": round(float(probs[2]) * 100, 2)
            },
            "sensor_values": sensor_display
        }
        predictions_list.append(entry)

    out_data = {
        "model": "federated_global_model",
        "test_subjects": test_subjects,
        "total_test_windows": len(predictions_list),
        "predictions": predictions_list
    }

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(out_data, f, indent=4)

    print(f"Successfully saved {len(predictions_list)} predictions to {output_json_path}")
    return out_data


if __name__ == "__main__":
    generate_dashboard_predictions()
