# flower_client.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
import flwr as fl
from collections import OrderedDict
import config
import warnings
from torchvision.models.resnet import BasicBlock, Bottleneck
from torch import Tensor


# --- HELPER FUNCTIONS TO FIX INPLACE OPS ---

def fix_all_inplace_activations(module):
    """
    Recursively finds all nn.ReLU, nn.ReLU6, and nn.Hardtanh
    and sets inplace=False.
    """
    for name, child in module.named_children():
        if isinstance(child, (nn.ReLU, nn.ReLU6, nn.Hardtanh)):
            if child.inplace:
                child.inplace = False
        else:
            fix_all_inplace_activations(child)
    return module

def _make_basic_block_forward(block_instance: BasicBlock):
    """ Creates a non-inplace forward method for a ResNet BasicBlock. """
    def new_forward(x: Tensor) -> Tensor:
        identity = x
        out = block_instance.conv1(x)
        out = block_instance.bn1(out)
        out = block_instance.relu(out)
        out = block_instance.conv2(out)
        out = block_instance.bn2(out)
        if block_instance.downsample is not None:
            identity = block_instance.downsample(x)
        
        out = out + identity
        
        out = block_instance.relu(out)
        return out
    return new_forward

def _make_bottleneck_forward(block_instance: Bottleneck):
    """ Creates a non-inplace forward method for a ResNet Bottleneck. """
    def new_forward(x: Tensor) -> Tensor:
        identity = x
        out = block_instance.conv1(x)
        out = block_instance.bn1(out)
        out = block_instance.relu(out)
        out = block_instance.conv2(x)
        out = block_instance.bn2(out)
        out = block_instance.relu(out)
        out = block_instance.conv3(x)
        out = block_instance.bn3(out)
        if block_instance.downsample is not None:
            identity = block_instance.downsample(x)
        
        out = out + identity
        
        out = block_instance.relu(out)
        return out
    return new_forward

def fix_resnet_skip_connections(module):
    """
    Recursively finds all ResNet blocks and patches their
    forward method to be non-inplace.
    """
    for name, child in module.named_children():
        if isinstance(child, BasicBlock):
            child.forward = _make_basic_block_forward(child)
        elif isinstance(child, Bottleneck):
            child.forward = _make_bottleneck_forward(child)
        else:
            fix_resnet_skip_connections(child)
    return module

# --- END HELPER FUNCTIONS ---


class DPFLClient(fl.client.NumPyClient):
    
    def __init__(self, cid, model, train_indices, train_dataset, noise_multiplier, max_grad_norm):
        self.cid = cid
        self.train_indices = train_indices
        self.train_dataset = train_dataset
        self.device = config.DEVICE
        self.noise_multiplier = noise_multiplier
        
        self.total_epsilon = 0.0
        self.round_num = 0

        self.model = model.to(self.device)
        self.privacy_engine = None
        self.dp_loader = None
        
        client_subset = Subset(self.train_dataset, self.train_indices)
        self.train_loader = DataLoader(
            client_subset,
            batch_size=config.LOCAL_BATCH_SIZE,
            shuffle=True,
            num_workers=0,
        )
        
        if self.noise_multiplier > 0:
            warnings.filterwarnings("ignore", message="Secure RNG turned off.")
            warnings.filterwarnings("ignore", message="Full backward hook is firing.*")

            print(f"Client {self.cid}: Fixing model BatchNorm for DP...")
            self.model = ModuleValidator.fix(self.model)
            
            print(f"Client {self.cid}: Fixing model inplace activations for DP...")
            self.model = fix_all_inplace_activations(self.model)
            
            print(f"Client {self.cid}: Fixing model ResNet skip connections for DP...")
            self.model = fix_resnet_skip_connections(self.model)
            
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.CLIENT_LR)
            
            self.privacy_engine = PrivacyEngine(accountant="rdp")
            
            self.model, self.optimizer, self.dp_loader = self.privacy_engine.make_private(
                module=self.model,
                optimizer=self.optimizer,
                data_loader=self.train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=max_grad_norm,
                poisson_sampling=True,
            )
        else:
            print(f"Client {self.cid}: Running WITHOUT privacy (noise=0.0).")
            self.optimizer = optim.SGD(self.model.parameters(), lr=config.CLIENT_LR)
            self.dp_loader = self.train_loader

    def get_parameters(self, config):
        """Return model parameters as a list of NumPy arrays."""
        if self.privacy_engine:
            return [val.cpu().numpy() for _, val in self.model._module.state_dict().items()]
        else:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters):
        """Set model parameters from a list of NumPy arrays."""
        if self.privacy_engine:
            keys = self.model._module.state_dict().keys()
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model._module.load_state_dict(state_dict, strict=True)
        else:
            keys = self.model.state_dict().keys()
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, server_config):
        """Train the model with Differential Privacy AND FedProx loss."""
        self.set_parameters(parameters)
        
        global_model_params = [p.clone().detach() for p in self.model.parameters()]
        
        self.model.train()
        
        loss_fn = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0
        
        local_epochs = server_config.get("local_epochs", config.LOCAL_EPOCHS)
        mu = server_config.get("proximal_mu", 0.1)
        
        for epoch in range(local_epochs):
            for data, target in self.dp_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                
                base_loss = loss_fn(output, target)
                prox_term = 0.0
                
                current_params = self.model._module.parameters() if self.privacy_engine else self.model.parameters()
                
                for global_p, current_p in zip(global_model_params, current_params):
                    prox_term += torch.sum(torch.pow(current_p - global_p, 2))
                
                loss = base_loss + (mu / 2.0) * prox_term
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        if self.privacy_engine:
            try:
                self.total_epsilon = self.privacy_engine.get_epsilon(delta=config.PRIVACY_DELTA)
            except Exception as e:
                print(f"Client {self.cid}: Warning - Could not compute epsilon: {e}")
        
        return (
            self.get_parameters(config={}),
            len(self.train_indices),
            {
                "loss": avg_loss,
                "epsilon": self.total_epsilon,
                "total_epsilon": self.total_epsilon,
            }
        )
    
    def evaluate(self, parameters, config):
        """Evaluate the model on local data (optional)."""
        self.set_parameters(parameters)
        return 0.0, 0, {"accuracy": 0.0}