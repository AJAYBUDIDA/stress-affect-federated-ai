"""
AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI
Centralized Model Training Pipeline with Subject-Wise Data Splitting
"""

import os
import sys
import random
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Add root directory to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import StressClassifier

# Target label conversion: original 1, 2, 3 -> 0, 1, 2
LABEL_TO_CLASS = {
    1: 0,  # Neutral
    2: 1,  # Stress
    3: 2   # Amusement
}


def set_seed(seed=42):
    """Sets fixed random seed for 100% reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_split_data(csv_path, train_subjs, val_subjs, test_subjs):
    """
    Loads WESAD processed features CSV and performs subject-wise split.
    Fits StandardScaler ONLY on training subjects.
    """
    df = pd.read_csv(csv_path)
    
    metadata_cols = ['subject_id', 'label', 'label_name', 'window_id']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    # Target encoding: 1->0, 2->1, 3->2
    df['target'] = df['label'].map(LABEL_TO_CLASS)
    
    # Filter datasets by subject IDs
    train_df = df[df['subject_id'].isin(train_subjs)].copy()
    val_df = df[df['subject_id'].isin(val_subjs)].copy()
    test_df = df[df['subject_id'].isin(test_subjs)].copy()
    
    X_train_raw = train_df[feature_cols].values
    y_train = train_df['target'].values
    
    X_val_raw = val_df[feature_cols].values
    y_val = val_df['target'].values
    
    X_test_raw = test_df[feature_cols].values
    y_test = test_df['target'].values
    
    # Standardize features using scaler fitted ONLY on training set
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)
    X_test = scaler.transform(X_test_raw)
    
    # Save scaler
    os.makedirs("models", exist_ok=True)
    scaler_path = os.path.join("models", "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved fitted StandardScaler to {scaler_path}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols, test_df


def train_centralized_model(csv_path, epochs=30, batch_size=64, lr=0.001):
    """
    Executes training loop for the centralized PyTorch model.
    """
    set_seed(42)
    
    # Subject-wise split definition across 15 subjects
    all_subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
    train_subjects = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13']  # 11 subjects (73.3%)
    val_subjects = ['S14', 'S15']                                                           # 2 subjects (13.3%)
    test_subjects = ['S16', 'S17']                                                         # 2 subjects (13.3%)
    
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te), feature_cols, test_df = load_and_split_data(
        csv_path, train_subjects, val_subjects, test_subjects
    )
    
    print("\n" + "="*50)
    print("SUBJECT-WISE DATASET SPLIT SUMMARY")
    print("="*50)
    print(f"Training subjects ({len(train_subjects)}): {train_subjects}")
    print(f"Validation subjects ({len(val_subjects)}): {val_subjects}")
    print(f"Test subjects ({len(test_subjects)}): {test_subjects}")
    print(f"Training samples: {len(X_tr):,}")
    print(f"Validation samples: {len(X_va):,}")
    print(f"Test samples: {len(X_te):,}")
    print("="*50 + "\n")
    
    # Convert arrays to PyTorch Tensors
    train_dataset = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_va, dtype=torch.float32), torch.tensor(y_va, dtype=torch.long))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize Model, Loss, Optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = StressClassifier(input_dim=len(feature_cols), num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print("Starting Centralized Model Training...")
    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct_tr = 0
        total_tr = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct_tr += (preds == targets).sum().item()
            total_tr += targets.size(0)
            
        train_loss = running_loss / total_tr
        train_acc = (correct_tr / total_tr) * 100.0
        
        # --- Validation Phase ---
        model.eval()
        val_running_loss = 0.0
        correct_va = 0
        total_va = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct_va += (preds == targets).sum().item()
                total_va += targets.size(0)
                
        val_loss = val_running_loss / total_va
        val_acc = (correct_va / total_va) * 100.0
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
    # Save Model Checkpoint
    model_path = os.path.join("models", "centralized_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\nSaved trained model checkpoint to {model_path}")
    
    # Save Training History Plot
    os.makedirs("results/plots", exist_ok=True)
    plot_path = os.path.join("results", "plots", "training_history.png")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', color='#2b5c8f', lw=2)
    plt.plot(range(1, epochs + 1), history['val_loss'], label='Val Loss', color='#d9534f', lw=2, linestyle='--')
    plt.title('Training & Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], label='Train Accuracy', color='#2b5c8f', lw=2)
    plt.plot(range(1, epochs + 1), history['val_acc'], label='Val Accuracy', color='#5cb85c', lw=2, linestyle='--')
    plt.title('Training & Validation Accuracy (%)', fontsize=12, fontweight='bold')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved training history plot to {plot_path}")
    
    return model, (X_te, y_te), test_subjects, test_df


if __name__ == "__main__":
    csv_file = os.path.join("data", "processed", "wesad_features.csv")
    train_centralized_model(csv_file, epochs=30, batch_size=64, lr=0.001)
