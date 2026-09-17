"""
AI-Based Real-Time Stress and Affect Detection Using Federated Learning and Explainable AI
Flower Federated Server Strategy & Metrics Aggregation
"""

import os
import sys
import matplotlib.pyplot as plt
import flwr as fl

# Add root directory to module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class SaveGlobalModelFedAvg(fl.server.strategy.FedAvg):
    """
    Custom FedAvg Strategy that captures and retains the aggregated global parameters after each round.
    Ensures final aggregated parameters can be saved to models/federated_global_model.pth
    and evaluated on unseen test subjects S16, S17 after Round 5.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            # Convert Flower Parameters object to list of NumPy ndarrays
            self.latest_parameters = fl.common.parameters_to_ndarrays(aggregated_parameters)
        return aggregated_parameters, metrics


def aggregate_fit_metrics(metrics):
    """Aggregates local training metrics across clients for each FL round."""
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}
    
    weighted_acc = sum(num_samples * m["train_acc"] for num_samples, m in metrics) / total_samples
    weighted_loss = sum(num_samples * m["train_loss"] for num_samples, m in metrics) / total_samples
    return {"train_acc": weighted_acc, "train_loss": weighted_loss}


def aggregate_eval_metrics(metrics):
    """Aggregates local evaluation metrics across clients."""
    total_samples = sum(num_samples for num_samples, _ in metrics)
    if total_samples == 0:
        return {}
        
    weighted_acc = sum(num_samples * m["val_acc"] for num_samples, m in metrics) / total_samples
    return {"val_acc": weighted_acc}


def get_fedavg_strategy(fraction_fit=1.0, fraction_evaluate=1.0, min_fit_clients=3, min_eval_clients=3, min_available_clients=3):
    """
    Constructs custom SaveGlobalModelFedAvg Strategy.
    """
    strategy = SaveGlobalModelFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_eval_clients,
        min_available_clients=min_available_clients,
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn=aggregate_eval_metrics
    )
    return strategy


def plot_federated_round_metrics(history, save_path="results/plots/federated_round_metrics.png"):
    """
    Plots training loss and accuracy history across Federated Learning rounds.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fit_acc = [val for _, val in history.metrics_distributed.get("train_acc", [])] if hasattr(history, 'metrics_distributed') and "train_acc" in history.metrics_distributed else []
    fit_loss = [val for _, val in history.metrics_distributed.get("train_loss", [])] if hasattr(history, 'metrics_distributed') and "train_loss" in history.metrics_distributed else []
    
    rounds_list = list(range(1, len(fit_acc) + 1))
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rounds_list, fit_loss, marker='o', color='#2b5c8f', lw=2, label='Aggregated Train Loss')
    plt.title('Federated Learning - Round Loss (FedAvg)', fontsize=12, fontweight='bold')
    plt.xlabel('FL Round')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(rounds_list, fit_acc, marker='s', color='#5cb85c', lw=2, label='Aggregated Train Accuracy (%)')
    plt.title('Federated Learning - Round Accuracy (FedAvg)', fontsize=12, fontweight='bold')
    plt.xlabel('FL Round')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved federated round metrics plot to {save_path}")
