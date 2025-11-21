# run_sensitivity_flower.py
# main script to run!
import asyncio
import numpy as np
import config
from main_flower import run_flower_experiment
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from datetime import datetime

# Architectures to test
ARCHITECTURES_TO_TEST = [ 'ResNet18', 'MobileNetV2','ViT','SimpleCNN']

# Sensitivity parameters
BASE_SIGMA = config.NOISE_MULTIPLIER_SIGMA
H = config.SIGMA_PERTURBATION_H
NUM_SEEDS = config.NUM_SENSITIVITY_SEEDS


async def run_sensitivity_for_arch(model_name, executor):
    """
    Calculates Architectural Sensitivity (Ψ_A) for a single architecture.
    
    Ψ_A = E[∂L/∂σ] ≈ E[(L(σ+h) - L(σ-h)) / (2h)]
    """
    print(f"\n{'#'*70}")
    print(f"# ARCHITECTURAL SENSITIVITY ANALYSIS: {model_name}")
    print(f"{'#'*70}\n")
    
    sigma_plus = BASE_SIGMA + H
    sigma_minus = max(0.1, BASE_SIGMA - H)
    
    losses_plus = []
    losses_minus = []
    accuracies_plus = []
    accuracies_minus = []
    
    # Get the current running event loop
    loop = asyncio.get_running_loop()
    
    for seed in range(NUM_SEEDS):
        # Set random seed for reproducibility
        np.random.seed(42 + seed)
        
        print(f"\n{'='*70}")
        print(f"Seed {seed+1}/{NUM_SEEDS}: Testing σ = {sigma_plus:.4f}")
        print(f"{'='*70}")
        
        loss_plus, acc_plus = await loop.run_in_executor(
            executor,
            run_flower_experiment, 
            model_name,
            sigma_plus
        )
        losses_plus.append(loss_plus)
        accuracies_plus.append(acc_plus)
        
        print(f"\n{'='*70}")
        print(f"Seed {seed+1}/{NUM_SEEDS}: Testing σ = {sigma_minus:.4f}")
        print(f"{'='*70}")
        
        loss_minus, acc_minus = await loop.run_in_executor(
            executor,
            run_flower_experiment,
            model_name,
            sigma_minus
        )
        losses_minus.append(loss_minus)
        accuracies_minus.append(acc_minus)
    
    # Calculate expectations
    avg_loss_plus = np.mean(losses_plus)
    avg_loss_minus = np.mean(losses_minus)
    std_loss_plus = np.std(losses_plus)
    std_loss_minus = np.std(losses_minus)
    
    avg_acc_plus = np.mean(accuracies_plus)
    avg_acc_minus = np.mean(accuracies_minus)
    
    # Calculate Ψ_A using finite difference
    psi_a = (avg_loss_plus - avg_loss_minus) / (2 * H)
    
    # Also calculate accuracy sensitivity
    acc_sensitivity = (avg_acc_minus - avg_acc_plus) / (2 * H)
    
    results = {
        'model_name': model_name,
        'psi_a': psi_a,
        'acc_sensitivity': acc_sensitivity,
        'sigma_plus': sigma_plus,
        'sigma_minus': sigma_minus,
        'avg_loss_plus': avg_loss_plus,
        'avg_loss_minus': avg_loss_minus,
        'std_loss_plus': std_loss_plus,
        'std_loss_minus': std_loss_minus,
        'avg_acc_plus': avg_acc_plus,
        'avg_acc_minus': avg_acc_minus,
        'losses_plus': losses_plus,
        'losses_minus': losses_minus,
        'accuracies_plus': accuracies_plus,
        'accuracies_minus': accuracies_minus
    }
    
    print(f"\n{'*'*70}")
    print(f"RESULTS FOR {model_name}:")
    print(f"{'*'*70}")
    print(f"Avg Loss (σ={sigma_plus:.4f}): {avg_loss_plus:.4f} ± {std_loss_plus:.4f}")
    print(f"Avg Loss (σ={sigma_minus:.4f}): {avg_loss_minus:.4f} ± {std_loss_minus:.4f}")
    print(f"Avg Accuracy (σ={sigma_plus:.4f}): {avg_acc_plus:.2f}%")
    print(f"Avg Accuracy (σ={sigma_minus:.4f}): {avg_acc_minus:.2f}%")
    print(f"\n Architectural Sensitivity (Ψ_A): {psi_a:.6f}")
    print(f"   (Lower = more robust to privacy noise)")
    print(f"   Accuracy Sensitivity: {acc_sensitivity:.6f} %/σ")
    print(f"{'*'*70}\n")
    
    return results


async def main(executor):
    """Run sensitivity analysis for all architectures."""
    all_results = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sensitivity_results_{timestamp}.json"
    print(f"Results will be saved incrementally to: {filename}\n")
    
    for arch in ARCHITECTURES_TO_TEST:
        results = await run_sensitivity_for_arch(arch, executor)
        all_results.append(results)
        
        try:
            with open(filename, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n Successfully saved intermediate results for {arch} to {filename}\n")
        except IOError as e:
            print(f"\n Error saving intermediate results for {arch}: {e}\n")
    
    # Print final comparison
    print(f"\n{'#'*70}")
    print(f"# FINAL ARCHITECTURAL SENSITIVITY COMPARISON")
    print(f"{'#'*70}\n")
    print(f"{'Model':<20} {'Ψ_A (Loss Sens.)':<20} {'Acc. Sens. (%/σ)':<20}")
    print(f"{'-'*60}")
    
    # Sort by Ψ_A (lower is better)
    sorted_results = sorted(all_results, key=lambda x: x['psi_a'])
    
    for r in sorted_results:
        print(f"{r['model_name']:<20} {r['psi_a']:<20.6f} {r['acc_sensitivity']:<20.6f}")
    
    print(f"\n{'='*70}")
    if sorted_results:
        print(f"Best Architecture (lowest Ψ_A): {sorted_results[0]['model_name']}")
        print(f"   Ψ_A = {sorted_results[0]['psi_a']:.6f}")
    else:
        print("No results to compare.")
    print(f"{'='*70}\n")
    
    print(f"Final results are saved in: {filename}")
    
    return all_results


if __name__ == "__main__":
    with ExitStack() as stack:
        # Use ThreadPool to run the blocking simulations in parallel
        # We limit max_workers to 2, assuming 1 GPU.
        # If you have more GPUs, you can increase this.
        max_parallel_runs = 2 
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=max_parallel_runs))
        print(f"Starting analysis with {max_parallel_runs} parallel runs.")
        asyncio.run(main(executor))