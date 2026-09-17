"""
AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI
Model Evaluation & Metrics Visualization Module
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Add root directory to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import StressClassifier
from src.train import LABEL_TO_CLASS

CLASS_NAMES = ['Neutral', 'Stress', 'Amusement']


def evaluate_centralized_model(csv_path):
    """
    Evaluates trained PyTorch model on unseen test subjects and generates metrics & plots.
    """
    df = pd.read_csv(csv_path)
    
    metadata_cols = ['subject_id', 'label', 'label_name', 'window_id']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    df['target'] = df['label'].map(LABEL_TO_CLASS)
    
    # Subject-wise dataset partitions
    train_subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13']
    val_subjects = ['S14', 'S15']
    test_subjects = ['S16', 'S17']
    
    train_df = df[df['subject_id'].isin(train_subjects)]
    val_df = df[df['subject_id'].isin(val_subjects)]
    test_df = df[df['subject_id'].isin(test_subjects)]
    
    # Load Scaler
    scaler_path = os.path.join("models", "scaler.pkl")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found at {scaler_path}. Run training first.")
        
    scaler = joblib.load(scaler_path)
    
    X_test_raw = test_df[feature_cols].values
    y_test = test_df['target'].values
    X_test = scaler.transform(X_test_raw)
    
    # Load Model
    model_path = os.path.join("models", "centralized_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Run training first.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StressClassifier(input_dim=len(feature_cols), num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Inference on Test Set
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_test_tensor)
        preds = logits.argmax(dim=1).cpu().numpy()
        
    # Calculate Overall & Per-Class Metrics
    acc = float(accuracy_score(y_test, preds))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average='macro')
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, preds, average='weighted')
    
    cm = confusion_matrix(y_test, preds)
    report_str = classification_report(y_test, preds, target_names=CLASS_NAMES, digits=4)
    
    # Save Confusion Matrix Plot
    os.makedirs("results/plots", exist_ok=True)
    cm_path = os.path.join("results", "plots", "confusion_matrix.png")
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    plt.title('Confusion Matrix - Centralized Baseline Model\n(Unseen Test Subjects S16, S17)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('True Ground Truth Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # Save Centralized Metrics JSON
    metrics_json_path = os.path.join("results", "centralized_metrics.json")
    metrics_dict = {
        "accuracy": acc,
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(prec_weighted),
        "recall_weighted": float(rec_weighted),
        "f1_weighted": float(f1_weighted),
        "test_subjects": test_subjects,
        "test_samples": len(y_test)
    }
    with open(metrics_json_path, 'w') as f:
        import json
        json.dump(metrics_dict, f, indent=4)
        
    return {
        'acc': acc,
        'prec_macro': prec_macro,
        'rec_macro': rec_macro,
        'f1_macro': f1_macro,
        'prec_weighted': prec_weighted,
        'rec_weighted': rec_weighted,
        'f1_weighted': f1_weighted,
        'cm': cm,
        'report': report_str
    }


if __name__ == "__main__":
    csv_file = os.path.join("data", "processed", "wesad_features.csv")
    res = evaluate_centralized_model(csv_file)
    print("Centralized Baseline Evaluation Result:", res)
