# config.py
import torch

# --- Simulation Parameters ---
NUM_CLIENTS = 10
NUM_ROUNDS = 30   
STALENESS_FUNCTION = lambda s: 1 / (1.0 + s)

# --- Model & Data Parameters ---
DATASET = 'CIFAR10'
NON_IID_ALPHA = 0.5      
LOCAL_BATCH_SIZE = 32    
LOCAL_EPOCHS = 1        

# --- DP Parameters (The Base) ---
PRIVACY_EPSILON = 10.0
PRIVACY_DELTA = 1e-5
MAX_GRAD_NORM = 1.0
NOISE_MULTIPLIER_SIGMA = 0.5

CLIENT_LR = 0.01
SERVER_LR = 1.0

# --- Device Config ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Sensitivity Parameters (The Slope Calculation) ---
SIGMA_PERTURBATION_H = 0.1 
NUM_SENSITIVITY_SEEDS = 3