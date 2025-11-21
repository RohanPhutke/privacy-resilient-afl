# Privacy-Resilient Architectures in Asynchronous Federated Learning

## Abstract

This repository contains the official implementation for the research project  **"Resilience in Differentially Private Asynchronous Federated Learning"** . This study investigates the under-explored relationship between neural network architectural choices and their robustness to Differential Privacy (DP) noise within a Federated Learning (FL) framework.

We propose a novel metric,  **Architectural Sensitivity (** $\Psi_A$ **)** , to quantify the stability of a model's learning process under varying privacy budgets. The findings demonstrate a significant divergence between computational efficiency and privacy efficiency, highlighting the need for co-designing architectures with privacy mechanisms.

## Table of Contents

* [Abstract](https://www.google.com/search?q=%23abstract "null")
* [Key Contributions](https://www.google.com/search?q=%23key-contributions "null")
* [Methodology](https://www.google.com/search?q=%23methodology "null")
* [Installation](https://www.google.com/search?q=%23installation "null")
* [Usage](https://www.google.com/search?q=%23usage "null")
* [Project Structure](https://www.google.com/search?q=%23project-structure "null")
* [Results](https://www.google.com/search?q=%23results "null")
* [License](https://www.google.com/search?q=%23license "null")

## Key Contributions

* **Architectural Sensitivity Metric:** Introduction of a mathematical framework ($\Psi_A$) to measure the gradient of performance degradation with respect to privacy noise.
* **Empirical Benchmarks:** Comprehensive evaluation of ResNet18, MobileNetV2, Vision Transformer (ViT), and SimpleCNN on Non-IID CIFAR-10 data.
* **DP-FedProx Implementation:** A robust implementation of the FedProx algorithm integrated with Opacus for client-side differential privacy.
* **Structural Analysis:** Detailed analysis of why specific components (e.g., Batch Normalization, Depthwise Separable Convolutions) fail or succeed in high-privacy regimes.

## Methodology

### Federated Learning Framework

The system is built using **Flower (flwr)** and  **Opacus** . It simulates a realistic Federated Learning environment with the following characteristics:

* **Algorithm:** DP-FedProx (Differential Privacy + Federated Proximal Optimization).
* **Data Distribution:** Non-IID partitioning of CIFAR-10 using a Dirichlet distribution ($\alpha=0.5$).
* **Privacy Mechanism:** Client-level DP-SGD with gradient clipping and Gaussian noise injection.

### Architectural Sensitivity ($\Psi_A$)

We define Architectural Sensitivity as the rate of change in global loss $\mathcal{L}$ with respect to the noise multiplier $\sigma$:

$$
\Psi_A = \mathbb{E} \left[ \frac{\partial \mathcal{L}_{global}}{\partial \sigma} \right]$$*  **Low ** $\Psi_A$**:** Indicates a robust architecture that maintains utility despite increased noise.
*  **High ** $\Psi_A$**:** Indicates a fragile architecture where performance degrades rapidly.

## Installation

### Prerequisites

* Python 3.9, 3.10, or 3.11
* CUDA-enabled GPU (Recommended)

### Setup

1. Clone the repository:
   ```
   git clone [https://github.com/RohanPhutke/privacy-resilient-afl.git](https://github.com/RohanPhutke/privacy-resilient-afl.git)
   cd privacy-resilient-afl

   ```
1. Create and activate a virtual environment (Recommended):
   ```
   # Using Conda
   conda create -n dp_fl_env python=3.11 -y
   conda activate dp_fl_env

   ```
1. Install dependencies:
   ```
   # Install PyTorch (Adjust command based on your CUDA version)
   pip3 install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

   # Install Project Requirements
   pip install flwr["simulation"] opacus timm numpy matplotlib seaborn

   ```

## Usage

### Configuration

All hyperparameters for the simulation, privacy budget, and model selection are defined in `config.py`. You can modify:

* `NUM_CLIENTS`: Number of participating clients.
* `NOISE_MULTIPLIER_SIGMA`: The base noise level ($\sigma$).
* `NON_IID_ALPHA`: The degree of data heterogeneity.

### Running Sanity Checks

Before running the full analysis, verify the setup with a single run (SimpleCNN, No Noise):

```
python main_flower.py

```

### Running the Full Sensitivity Analysis

To execute the complete experiment, which evaluates all architectures across multiple noise regimes and calculates $\Psi_A$:

```
python run_sensitivity_analysis.py

```

This script will:

1. Train ResNet18, ViT, and SimpleCNN across defined noise levels.
1. Calculate the sensitivity metrics.
1. Save the results incrementally to a JSON file.

## Project Structure

```
.
├── config.py                     # Centralized configuration file
├── data_setup.py                 # Data downloading and Non-IID partitioning
├── flower_client.py              # FL Client implementation (DP + FedProx logic)
├── flower_server.py              # Server-side evaluation and aggregation
├── main_flower.py                # Single experiment runner
├── models.py                     # Model architectures (with DP compatibility fixes)
├── run_sensitivity_analysis.py   # Main script for sensitivity experiments
└── README.md                     # Project documentation

```
$$
