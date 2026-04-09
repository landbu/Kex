import torch
import torch.nn as nn

from architectures.networkArchitectures import networkRegistry

def loadNetwork(name: str) -> nn.Module:
    NetworkClass = networkRegistry[name]
    network = NetworkClass()
    network.load_state_dict(state_dict = torch.load(f"networks/{name}.pth"))
    network.eval()
    return network