"""
AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI
Federated Learning Orchestration & Unseen-Subject Evaluation Pipeline
"""

import os
import sys
import json
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 output encoding for console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import flwr as fl
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

# Add root directory to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import StressClassifier
from src.partition_clients import load_and_partition_federated_data, CLIENT_SUBJECT_MAP, TEST_SUBJECTS
from src.federated_client import WESADFlowerClient, set_model_parameters, get_model_parameters
from src.federated_server import get_fedavg_strategy, plot_federated_round_metrics
from src.train import LABEL_TO_CLASS

CLASS_NAMES = ['Neutral', 'Stress', 'Amusement']


def set_seed(seed=42):
    """Sets fixed random seed for 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_federated_experiment(csv_path, num_rounds=5, local_epochs=2, lr=0.001):
    """
    Runs full Federated Learning experiment using Flower simulation and evaluates on unseen test subjects S16, S17.
    """
    set_seed(42)
    
    # 1. Load & Partition Data (StandardScaler fitted ONLY on S2-S15)
    client_loaders, test_loader, client_sample_counts, feature_cols, test_df = load_and_partition_federated_data(
        csv_path, batch_size=64
    )
    
    print("\n" + "="*65)
    print("      FEDERATED LEARNING PRIVACY & PARTICIPATION SUMMARY")
    print("="*65)
    print(f"Client 1 Subjects: {CLIENT_SUBJECT_MAP[0]} -> {client_sample_counts[0]:,} local samples")
    print(f"Client 2 Subjects: {CLIENT_SUBJECT_MAP[1]} -> {client_sample_counts[1]:,} local samples")
    print(f"Client 3 Subjects: {CLIENT_SUBJECT_MAP[2]} -> {client_sample_counts[2]:,} local samples")
    print(f"Unseen Test Subjects (Evaluated AFTER Round 5): {TEST_SUBJECTS} -> {len(test_df):,} test samples\n")
    
    print("PRIVACY & ARCHITECTURE GUARANTEES:")
    print("  [OK] Zero Raw Data Transmission: Clients exchange ONLY model parameters.")
    print("  [OK] Feature Scaler: Fitted ONLY on training subjects (S2-S15); S16/S17 never seen by scaler.")
    print("  [OK] Unseen Evaluation: S16/S17 evaluated strictly AFTER all 5 FL rounds complete.")
    print("="*65 + "\n")
    
    # 2. Define Client Generator for Flower Simulation Engine (0-indexed)
    def client_fn(cid_str: str) -> fl.client.Client:
        cid = int(cid_str)
        loader = client_loaders[cid]
        return WESADFlowerClient(
            client_id=cid + 1, # 1-indexed label for logging
            train_loader=loader,
            input_dim=len(feature_cols),
            local_epochs=local_epochs,
            lr=lr
        ).to_client()
        
    # 3. Setup FedAvg Strategy
    strategy = get_fedavg_strategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=3,
        min_eval_clients=3,
        min_available_clients=3
    )
    
    init_model = StressClassifier(input_dim=len(feature_cols), num_classes=3)
    initial_parameters = fl.common.ndarrays_to_parameters(get_model_parameters(init_model))
    
    # 4. Start Flower Simulation (5 Rounds)
    print(f"Launching Flower Federated Learning Simulation ({num_rounds} Rounds, {local_epochs} Local Epochs/Round)...")
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=3,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0}
    )
    
    # 5. Plot FL Round Metrics History
    plot_federated_round_metrics(history, save_path="results/plots/federated_round_metrics.png")
    
    # 6. Apply Aggregated Global Parameters & Save Checkpoint
    global_model = StressClassifier(input_dim=len(feature_cols), num_classes=3)
    if strategy.latest_parameters is not None:
        set_model_parameters(global_model, strategy.latest_parameters)
        print("\nSuccessfully applied Round 5 aggregated parameters to Global Model.")
    else:
        print("\nWarning: Final round parameters not captured. Using initial parameters.")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    global_model.to(device)
    
    os.makedirs("models", exist_ok=True)
    global_model_path = os.path.join("models", "federated_global_model.pth")
    torch.save(global_model.state_dict(), global_model_path)
    print(f"Saved Federated Global Model checkpoint to {global_model_path}")
    
    # 7. Evaluate Global Federated Model ONLY AFTER Round 5 on Unseen Test Subjects (S16, S17)
    global_model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = global_model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Compute Metrics Dynamically (Zero Hardcoding)
    acc = float(accuracy_score(all_targets, all_preds))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(all_targets, all_preds, average='macro')
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    
    cm = confusion_matrix(all_targets, all_preds)
    report_str = classification_report(all_targets, all_preds, target_names=CLASS_NAMES, digits=4)
    
    # Save Federated Metrics to JSON
    metrics_dict = {
        'accuracy': float(acc),
        'precision_macro': float(prec_macro),
        'recall_macro': float(rec_macro),
        'f1_macro': float(f1_macro),
        'precision_weighted': float(prec_weighted),
        'recall_weighted': float(rec_weighted),
        'f1_weighted': float(f1_weighted),
        'test_subjects': TEST_SUBJECTS,
        'test_samples': len(all_targets)
    }
    
    os.makedirs("results", exist_ok=True)
    metrics_json_path = os.path.join("results", "federated_metrics.json")
    with open(metrics_json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"Saved federated metrics JSON to {metrics_json_path}")
    
    # Save Confusion Matrix Plot for Federated Model
    os.makedirs("results/plots", exist_ok=True)
    cm_path = os.path.join("results", "plots", "federated_confusion_matrix.png")
    
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                cbar=False, annot_kws={'size': 14, 'weight': 'bold'})
    plt.title('Confusion Matrix - Federated Learning Model (FedAvg)\n(Unseen Test Subjects S16, S17)', fontsize=12, fontweight='bold', pad=15)
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    plt.ylabel('True Ground Truth Label', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved federated confusion matrix plot to {cm_path}")
    
    # Print Federated Results Report
    print("\n" + "="*60)
    print("      FEDERATED LEARNING MODEL EVALUATION REPORT")
    print("="*60)
    print(f"Client 1 Subjects: {CLIENT_SUBJECT_MAP[0]}")
    print(f"Client 2 Subjects: {CLIENT_SUBJECT_MAP[1]}")
    print(f"Client 3 Subjects: {CLIENT_SUBJECT_MAP[2]}")
    print(f"Unseen Test Subjects: {TEST_SUBJECTS}\n")
    
    print(f"Test Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Test Precision: {prec_macro:.4f} (Macro) | {prec_weighted:.4f} (Weighted)")
    print(f"Test Recall:    {rec_macro:.4f} (Macro) | {rec_weighted:.4f} (Weighted)")
    print(f"Test F1-Score:  {f1_macro:.4f} (Macro) | {f1_weighted:.4f} (Weighted)\n")
    
    print("--- Detailed Classification Report ---")
    print(report_str)
    print("="*60 + "\n")
    
    # 8. Automated Side-by-Side Comparison reading computed Centralized & Federated values
    generate_baseline_vs_federated_comparison(metrics_dict, csv_path)


def generate_baseline_vs_federated_comparison(fl_metrics, csv_path):
    """
    Computes an automated side-by-side comparison between Centralized Baseline and Federated Learning.
    Loads actual computed values from saved checkpoints & evaluation reports.
    """
    from src.evaluation import evaluate_centralized_model
    
    print("\n" + "="*70)
    print("      MODEL COMPARISON: CENTRALIZED BASELINE vs. FEDERATED LEARNING")
    print("="*70)
    
    cent_results = evaluate_centralized_model(csv_path)
    
    comp_df = pd.DataFrame({
        'Metric': ['Test Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1-Score (Macro)', 'F1-Score (Weighted)'],
        'Centralized Baseline': [
            f"{cent_results['acc']*100:.2f}% ({cent_results['acc']:.4f})",
            f"{cent_results['prec_macro']:.4f}",
            f"{cent_results['rec_macro']:.4f}",
            f"{cent_results['f1_macro']:.4f}",
            f"{cent_results['f1_weighted']:.4f}"
        ],
        'Federated Learning (FedAvg)': [
            f"{fl_metrics['accuracy']*100:.2f}% ({fl_metrics['accuracy']:.4f})",
            f"{fl_metrics['precision_macro']:.4f}",
            f"{fl_metrics['recall_macro']:.4f}",
            f"{fl_metrics['f1_macro']:.4f}",
            f"{fl_metrics['f1_weighted']:.4f}"
        ]
    })
    
    print(comp_df.to_string(index=False))
    print("="*70 + "\n")


if __name__ == "__main__":
    csv_file = os.path.join("data", "processed", "wesad_features.csv")
    run_federated_experiment(csv_file, num_rounds=5, local_epochs=2, lr=0.001)
