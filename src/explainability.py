"""
AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI
Explainable AI (XAI) Module using SHAP for Federated PyTorch Model
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

import torch
from sklearn.preprocessing import StandardScaler

# Add root directory to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import StressClassifier
from src.train import LABEL_TO_CLASS

CLASS_NAMES = ['Neutral', 'Stress', 'Amusement']


def load_federated_model_and_scaler(feature_dim=68):
    """
    Loads trained Federated Global PyTorch model and fitted StandardScaler.
    """
    model_path = os.path.join("models", "federated_global_model.pth")
    scaler_path = os.path.join("models", "federated_scaler.pkl")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Federated model checkpoint not found at {model_path}. Run FL training first.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Federated scaler file not found at {scaler_path}. Run FL partitioning first.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StressClassifier(input_dim=feature_dim, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    scaler = joblib.load(scaler_path)
    return model, scaler, device


def predict_probabilities(x_numpy, model, device):
    """
    Wrapper function to compute softmax class probabilities from PyTorch model for SHAP explainer.
    Input: NumPy array of shape (N, 68)
    Output: Softmax probability matrix of shape (N, 3)
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.tensor(x_numpy, dtype=torch.float32).to(device)
        logits = model(x_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
    return probs


def run_explainability_pipeline(csv_path):
    """
    Generates SHAP feature explanations on unseen test subjects S16 and S17.
    Background dataset is sampled STRICTLY from training subjects S2-S15.
    """
    df = pd.read_csv(csv_path)
    
    metadata_cols = ['subject_id', 'label', 'label_name', 'window_id']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    df['target'] = df['label'].map(LABEL_TO_CLASS)
    
    train_subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15']
    test_subjects = ['S16', 'S17']
    
    train_df = df[df['subject_id'].isin(train_subjects)].copy()
    test_df = df[df['subject_id'].isin(test_subjects)].copy()
    
    print("="*65)
    print("      EXPLAINABLE AI (XAI) PIPELINE WITH SHAP")
    print("="*65)
    print(f"Target Model: models/federated_global_model.pth")
    print(f"Target Scaler: models/federated_scaler.pkl")
    print(f"Training Background Subjects (S2-S15): {len(train_df):,} samples")
    print(f"Unseen Test Subjects (S16, S17): {len(test_df):,} test samples across 68 features\n")
    
    # Load Model and Scaler
    model, scaler, device = load_federated_model_and_scaler(feature_dim=len(feature_cols))
    
    # Construct background reference set strictly from TRAINING subjects S2-S15
    np.random.seed(42)
    bg_raw = train_df[feature_cols].values
    bg_indices = np.random.choice(len(bg_raw), size=min(100, len(bg_raw)), replace=False)
    bg_data_scaled = scaler.transform(bg_raw[bg_indices])
    
    # Unseen test dataset S16, S17 samples being explained
    X_test_raw = test_df[feature_cols].values
    X_test_scaled = scaler.transform(X_test_raw)
    
    eval_indices = np.random.choice(len(X_test_scaled), size=min(200, len(X_test_scaled)), replace=False)
    eval_data_scaled = X_test_scaled[eval_indices]
    eval_raw = X_test_raw[eval_indices]
    eval_df = test_df.iloc[eval_indices].copy()
    
    print(f"Methodology Check:")
    print(f"  [OK] SHAP Background Reference Set: {len(bg_data_scaled)} samples (Sampled ONLY from S2-S15)")
    print(f"  [OK] SHAP Test Explanation Set: {len(eval_data_scaled)} samples (Sampled ONLY from S16, S17)")
    print("Computing SHAP values...")
    
    def proba_fn(x):
        return predict_probabilities(x, model, device)
        
    explainer = shap.Explainer(proba_fn, bg_data_scaled)
    shap_explanation = explainer(eval_data_scaled)
    
    # Normalize SHAP output shape across SHAP versions into 3D array: (N_samples, 68_features, 3_classes)
    if isinstance(shap_explanation, list):
        shap_vals_3d = np.stack(shap_explanation, axis=-1)
    elif hasattr(shap_explanation, 'values'):
        raw_vals = shap_explanation.values
        if len(raw_vals.shape) == 3:
            shap_vals_3d = raw_vals
        elif len(raw_vals.shape) == 2:
            shap_vals_3d = np.expand_dims(raw_vals, axis=-1)
        else:
            shap_vals_3d = raw_vals
    else:
        raw_vals = np.array(shap_explanation)
        shap_vals_3d = raw_vals if len(raw_vals.shape) == 3 else np.expand_dims(raw_vals, axis=-1)
        
    print(f"\n[OK] Verified SHAP values 3D tensor shape: {shap_vals_3d.shape} (Samples, Features, Classes)")
    
    # --- 1. Save SHAP Summary Plot (results/xai/shap_summary.png) ---
    os.makedirs("results/xai", exist_ok=True)
    summary_plot_path = os.path.join("results", "xai", "shap_summary.png")
    
    plt.figure(figsize=(10, 8))
    # Pass list of 2D arrays for each class [ (N, 68), (N, 68), (N, 68) ]
    shap_list_for_plot = [shap_vals_3d[:, :, c] for c in range(3)]
    shap.summary_plot(
        shap_list_for_plot,
        features=eval_raw,
        feature_names=feature_cols,
        class_names=CLASS_NAMES,
        plot_type="bar",
        max_display=15,
        show=False
    )
    plt.title('SHAP Global Feature Importance across Stress & Affect Classes\n(Federated Global Model on Unseen Test Subjects S16, S17)', fontsize=11, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(summary_plot_path, dpi=300)
    plt.close()
    print(f"Saved SHAP summary plot to {summary_plot_path}")
    
    # --- 2. Save Feature Importance CSV (results/xai/shap_feature_importance.csv) ---
    # Mean absolute SHAP value across samples and classes for each feature
    mean_abs_shap = np.mean(np.abs(shap_vals_3d), axis=(0, 2))
    
    importance_df = pd.DataFrame({
        'feature_name': feature_cols,
        'mean_absolute_shap_value': mean_abs_shap
    }).sort_values(by='mean_absolute_shap_value', ascending=False).reset_index(drop=True)
    
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    csv_out_path = os.path.join("results", "xai", "shap_feature_importance.csv")
    importance_df.to_csv(csv_out_path, index=False)
    print(f"Saved feature importance CSV to {csv_out_path}")
    
    # --- 3. Save Local Explanation JSON (results/xai/sample_explanation.json) ---
    sample_idx = 0
    sample_features_scaled = eval_data_scaled[sample_idx:sample_idx+1]
    sample_probs = predict_probabilities(sample_features_scaled, model, device)[0]
    
    pred_class_idx = int(np.argmax(sample_probs))
    pred_class_name = CLASS_NAMES[pred_class_idx]
    pred_confidence = float(sample_probs[pred_class_idx])
    
    true_class_idx = int(eval_df.iloc[sample_idx]['target'])
    true_class_name = CLASS_NAMES[true_class_idx]
    sample_subj_id = str(eval_df.iloc[sample_idx]['subject_id'])
    
    sample_shap_for_pred = shap_vals_3d[sample_idx, :, pred_class_idx]
    
    feature_contribs = []
    for f_name, f_val, s_val in zip(feature_cols, eval_raw[sample_idx], sample_shap_for_pred):
        feature_contribs.append({
            'feature': f_name,
            'feature_value': float(f_val),
            'shap_value': float(s_val)
        })
        
    pos_contribs = sorted([fc for fc in feature_contribs if fc['shap_value'] > 0], key=lambda x: x['shap_value'], reverse=True)[:5]
    neg_contribs = sorted([fc for fc in feature_contribs if fc['shap_value'] < 0], key=lambda x: x['shap_value'])[:5]
    
    sample_explanation = {
        'subject_id': sample_subj_id,
        'true_label': true_class_name,
        'predicted_label': pred_class_name,
        'prediction_confidence': round(pred_confidence, 4),
        'top_positive_features': pos_contribs,
        'top_negative_features': neg_contribs
    }
    
    json_out_path = os.path.join("results", "xai", "sample_explanation.json")
    with open(json_out_path, 'w') as f:
        json.dump(sample_explanation, f, indent=4)
    print(f"Saved sample local explanation JSON to {json_out_path}\n")
    
    # --- 4. Print Summary Report ---
    print("="*60)
    print("      EXPLAINABLE AI (XAI) SUMMARY REPORT")
    print("="*60)
    print(f"Number of test samples evaluated with SHAP: {len(eval_data_scaled)}")
    print(f"Sample Subject ID: {sample_subj_id}")
    print(f"Sample True Label: {true_class_name}")
    print(f"Sample Predicted Class: {pred_class_name}")
    print(f"Sample Prediction Confidence: {pred_confidence*100:.2f}%\n")
    
    print("--- Top 10 Most Important Features (Mean |SHAP| Value) ---")
    for _, r in importance_df.head(10).iterrows():
        print(f"  Rank {r['rank']:2d}: {r['feature_name']:<20} | Mean |SHAP|: {r['mean_absolute_shap_value']:.6f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    csv_file = os.path.join("data", "processed", "wesad_features.csv")
    run_explainability_pipeline(csv_file)
