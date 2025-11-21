# flower_server.py

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict
import flwr as fl
from flwr.common import Metrics, Parameters, Scalar, parameters_to_ndarrays
import config
import numpy as np


def get_evaluate_fn(model, test_loader, model_name: str):
    """Return an evaluation function for server-side evaluation."""
    
    best_accuracy = 0.0
    
    def evaluate(server_round: int, parameters_ndarrays: fl.common.NDArrays, config_dict: Dict[str, Scalar]):
        """Evaluate global model on centralized test set."""
        
        nonlocal best_accuracy 
        params_dict = zip(model.state_dict().keys(), parameters_ndarrays)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.to(config.DEVICE)
        model.eval()
        
        loss_fn = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        total_loss = 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(config.DEVICE), target.to(config.DEVICE)
                output = model(data)
                total_loss += loss_fn(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(test_loader)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_path = f"best_model_{model_name}_sigma_{config.NOISE_MULTIPLIER_SIGMA}.pth"
            print(f"\nNew best accuracy: {accuracy:.2f}%! Saving model to {save_path}\n")
            torch.save(model.state_dict(), save_path)

        return avg_loss, {"accuracy": accuracy}
    
    return evaluate