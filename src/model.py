"""
AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI
Centralized PyTorch Neural Network Architecture
"""

import torch
import torch.nn as nn

class StressClassifier(nn.Module):
    """
    Feedforward Neural Network for 3-class Stress & Affect Detection:
    - Input: 68 numerical features
    - Linear(68, 32) -> ReLU -> Dropout(0.2)
    - Linear(32, 16) -> ReLU
    - Linear(16, 3) -> Output logits (Neutral=0, Stress=1, Amusement=2)
    """
    def __init__(self, input_dim=68, num_classes=3):
        super(StressClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )

    def forward(self, x):
        return self.network(x)
