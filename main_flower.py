# main_flower.py
import flwr as fl
import config
import data_setup
import models
from flower_client import DPFLClient
from collections import OrderedDict
from flwr.server.strategy import FedProx  
from flower_server import get_evaluate_fn 
from flwr.common import Metrics  # <-- ADD THIS
from typing import List, Tuple  # <-- ADD THIS
from opacus.validators import ModuleValidator

def aggregate_fit_metrics(results: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregates the 'loss' and 'total_epsilon' from clients."""
    if not results:
        return {}
    
    # Get metrics from all clients
    all_metrics = [metrics for num_examples, metrics in results]
    
    # Average the client 'loss'
    avg_loss = sum(m["loss"] for m in all_metrics) / len(all_metrics)
    
    # Get the max 'total_epsilon'
    max_epsilon = max(m["total_epsilon"] for m in all_metrics)
    
    # Return a neat dictionary
    return {"avg_client_loss": avg_loss, "max_client_epsilon": max_epsilon}

def run_flower_experiment(model_name, noise_multiplier):
    """
    Run a complete DP-AFL experiment using Flower.
    
    Args:
        model_name: Architecture to use (SimpleCNN, ResNet18, MobileNetV2)
        noise_multiplier: Sigma for DP-SGD noise
    
    Returns:
        final_loss, final_accuracy
    """
    config.NOISE_MULTIPLIER_SIGMA = noise_multiplier
    print(f"\n{'='*70}")
    print(f"Starting Flower DP-AFL Experiment")
    print(f"Model: {model_name}, σ={noise_multiplier}")
    print(f"{'='*70}\n")
    
    # Setup data
    train_ds, test_ds, num_classes = data_setup.get_data(config.DATASET)
    client_indices_list = data_setup.get_non_iid_indices(
        train_ds,
        config.NUM_CLIENTS,
        config.NON_IID_ALPHA
    )
    test_loader = data_setup.get_test_loader(test_ds)
    
    # Create global model
    model = models.get_model(model_name, num_classes, config.DATASET)

    # model structure that the clients will use.
    if noise_multiplier > 0:
        print("Fixing initial server model for DP compatibility...")
        model = ModuleValidator.fix(model)
    
    # Get initial parameters
    initial_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    
    strategy = FedProx(
        proximal_mu=0.1,
        
        evaluate_fn=get_evaluate_fn(model, test_loader,model_name), 
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        
        fraction_fit=1.0, 
        fraction_evaluate=0.0, 
        min_fit_clients=config.NUM_CLIENTS,
        min_evaluate_clients=0,
        min_available_clients=config.NUM_CLIENTS,
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters),
    )
    
    def client_fn(cid: str):
        """Create a Flower client."""
        client_id = int(cid)
        train_indices = client_indices_list[client_id]
        
        client = DPFLClient(
            cid=cid,
            model=models.get_model(model_name, num_classes, config.DATASET),
            train_indices=train_indices,
            train_dataset=train_ds,
            noise_multiplier=noise_multiplier,
            max_grad_norm=config.MAX_GRAD_NORM
        )
        return client.to_client()
    

    # Configure client resources
    client_resources = {
        "num_cpus": 1,
        "num_gpus": 0.5,
    }
    
    # Start simulation
    print(f"Starting Flower simulation with {config.NUM_CLIENTS} clients")
    print(f"Rounds: {config.NUM_ROUNDS}")
    print(f"Local epochs: {config.LOCAL_EPOCHS}")
    print(f"Batch size: {config.LOCAL_BATCH_SIZE}\n")
    
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config.NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=config.NUM_ROUNDS),
        strategy=strategy,
        client_resources=client_resources,
    )
    
    # Extract final results
    final_loss = history.losses_centralized[-1][1] if history.losses_centralized else float('inf')
    final_accuracy = history.metrics_centralized["accuracy"][-1][1] if "accuracy" in history.metrics_centralized else 0.0
    
    # Print privacy summary
    all_epsilons = history.metrics_distributed_fit.get("total_epsilon", [])

    if all_epsilons:
        print(f"\n{'='*70}")
        print(f"PRIVACY BUDGET SUMMARY")
        print(f"{'='*70}")
        
        # Get all epsilon values from the last round
        last_round_num = all_epsilons[-1][0]
        last_round_epsilons = [
            eps for rnd, eps in all_epsilons if rnd == last_round_num
        ]
        
        max_epsilon = max(last_round_epsilons)
        avg_epsilon = sum(last_round_epsilons) / len(last_round_epsilons)
        
        print(f"Max ε per client (at round {last_round_num}): {max_epsilon:.4f}")
        print(f"Avg ε per client (at round {last_round_num}): {avg_epsilon:.4f}")

        print(f"δ: {config.PRIVACY_DELTA}")
        print(f"{'='*70}\n")
    else:
        # This will print if no privacy metrics were found
        print("\nPrivacy metrics (total_epsilon) not found in history.\n")
    
    return final_loss, final_accuracy


if __name__ == "__main__":
    config.MODEL_NAME = 'ResNet18'
    
    print(f"--- STARTING FINAL TEST ---")
    print(f"Model: {config.MODEL_NAME}")
    print(f"Noise: 0.5 (DP is ON)")
    print(f"Strategy: FedProx (mu=0.1)")
    
    loss, acc = run_flower_experiment(
        model_name=config.MODEL_NAME,
        noise_multiplier=0.5
    )
    
    print("--- FINAL TEST COMPLETE ---")