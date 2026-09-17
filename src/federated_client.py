"""
AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI
Flower NumPyClient Implementation

PRIVACY & DATA TRANSMISSION GUARANTEES:
1. No raw feature rows or training dataset samples are ever transmitted over Flower.
2. Local training is performed strictly inside WESADFlowerClient.fit().
3. Only model parameters (weights/biases arrays) and local sample counts are returned to the server.
"""

import os
import sys
from collections import OrderedDict
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl

# Add root directory to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model import StressClassifier


def get_model_parameters(model):
    """Extracts PyTorch model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_model_parameters(model, parameters):
    """Sets PyTorch model parameters from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


class WESADFlowerClient(fl.client.NumPyClient):
    """
    Flower Client for local decentralized PyTorch training on subject-partitioned data.
    """
    def __init__(self, client_id, train_loader, input_dim=68, local_epochs=2, lr=0.001):
        self.client_id = client_id
        self.train_loader = train_loader
        self.local_epochs = local_epochs
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = StressClassifier(input_dim=input_dim, num_classes=3).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def get_parameters(self, config):
        """Returns current local PyTorch model parameters to Flower framework."""
        return get_model_parameters(self.model)

    def fit(self, parameters, config):
        """
        Executes local PyTorch training for 2 epochs using parameters sent by server.
        Returns ONLY updated model weights, sample count, and training loss/acc metrics.
        NO raw feature samples are ever transmitted.
        """
        set_model_parameters(self.model, parameters)
        
        self.model.train()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        for epoch in range(self.local_epochs):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total_samples += targets.size(0)
                
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = (correct / total_samples) * 100.0 if total_samples > 0 else 0.0
        
        print(f"  [Client {self.client_id}] Local FL Training: {self.local_epochs} Epochs | Loss: {avg_loss:.4f} | Acc: {accuracy:.2f}% ({len(self.train_loader.dataset)} local samples)")
        
        # Returns ONLY model weights and local sample count
        return get_model_parameters(self.model), len(self.train_loader.dataset), {"train_loss": avg_loss, "train_acc": accuracy}

    def evaluate(self, parameters, config):
        """
        Evaluates current global model parameters on local client dataset.
        """
        set_model_parameters(self.model, parameters)
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total_samples += targets.size(0)
                
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = correct / total_samples if total_samples > 0 else 0.0
        
        return avg_loss, total_samples, {"val_acc": accuracy}
