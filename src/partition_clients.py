"""
AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI
Client Partitioning & Feature Scaling Setup
"""

import os
import sys
import joblib
import pandas as pd

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Add root directory to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.train import LABEL_TO_CLASS

# Subject partitioning across 3 simulated Federated Learning clients (0-indexed for Flower)
CLIENT_SUBJECT_MAP = {
    0: ['S2', 'S3', 'S4', 'S5', 'S6'],      # Client 0: 5 subjects
    1: ['S7', 'S8', 'S9', 'S10', 'S11'],    # Client 1: 5 subjects
    2: ['S13', 'S14', 'S15']                 # Client 2: 3 subjects
}

# Completely unseen test subjects for final global evaluation ONLY (after all 5 rounds)
TEST_SUBJECTS = ['S16', 'S17']


def load_and_partition_federated_data(csv_path, batch_size=64):
    """
    Loads processed WESAD features, partitions data by subject across 3 FL clients,
    fits a common StandardScaler ONLY on training subjects (S2-S15), and returns PyTorch DataLoaders.
    """
    df = pd.read_csv(csv_path)
    
    metadata_cols = ['subject_id', 'label', 'label_name', 'window_id']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    
    df['target'] = df['label'].map(LABEL_TO_CLASS)
    
    # All training subjects across 3 clients (S2-S15)
    all_train_subjects = [subj for subjs in CLIENT_SUBJECT_MAP.values() for subj in subjs]
    train_df = df[df['subject_id'].isin(all_train_subjects)].copy()
    
    # Fit StandardScaler ONLY on training subjects (S2-S15). S16 and S17 are NEVER seen by the scaler.
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols].values)
    
    os.makedirs("models", exist_ok=True)
    scaler_path = os.path.join("models", "federated_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"Saved Federated StandardScaler (fitted ONLY on S2-S15) to {scaler_path}")
    
    client_loaders = {}
    client_sample_counts = {}
    
    # Build local PyTorch DataLoader for each simulated client
    for client_id, subjs in CLIENT_SUBJECT_MAP.items():
        c_df = df[df['subject_id'].isin(subjs)].copy()
        X_raw = c_df[feature_cols].values
        y = c_df['target'].values
        
        X_scaled = scaler.transform(X_raw)
        
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        ds = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
        
        client_loaders[client_id] = loader
        client_sample_counts[client_id] = len(c_df)
        
    # Unseen Test set DataLoader (S16, S17) - Evaluated ONLY after all 5 FL rounds complete
    test_df = df[df['subject_id'].isin(TEST_SUBJECTS)].copy()
    X_test_raw = test_df[feature_cols].values
    y_test = test_df['target'].values
    X_test_scaled = scaler.transform(X_test_raw)
    
    test_ds = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return client_loaders, test_loader, client_sample_counts, feature_cols, test_df
