from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Type
import random
import numpy as np

@dataclass
class NetworkEntry:
    NetworkClass: Type[nn.Module]
    tags: set[str]

networkRegistry: dict[str, NetworkEntry] = {}

def registerNetwork(*tags: str):
    def registerNetworkInner(NetworkClass: Type[nn.Module]):
        name = NetworkClass.__name__
        networkRegistry[name] = NetworkEntry(NetworkClass=NetworkClass, tags=set(tags))
        return NetworkClass
    return registerNetworkInner


def make_registered_mlp(name, width, depth):
    class GeneratedMLP(nn.Module):
        def __init__(self):
            super().__init__()

            self.dense1 = nn.Linear(28 * 28, width)
            self.relu1 = nn.ReLU()

            for i in range(2, depth + 1):
                setattr(self, f"dense{i}", nn.Linear(width, width))
                setattr(self, f"relu{i}", nn.ReLU())

            setattr(self, f"dense{depth+1}", nn.Linear(width, 10))

        def forward(self, x: torch.Tensor):
            x = x.view(-1, 1 * 28 * 28)

            x = self.relu1(self.dense1(x))

            for i in range(2, depth + 1):
                dense = getattr(self, f"dense{i}")
                relu = getattr(self, f"relu{i}")
                x = relu(dense(x))

            x = getattr(self, f"dense{depth+1}")(x)

            return x
    
    network_is = "big" if width*depth>100 else "small"
    GeneratedMLP.__name__ = f"Dense{depth}x{width}"
    registerNetwork("dense", "constantWidth", name, network_is)(GeneratedMLP)

    return GeneratedMLP


def register_all_networks():    
    networks = [(1,10),(1,15),(1,20),(1,25),(1,30),(1,35),(1,40),(1,45),(1,50),(1,55),(1,60),(1,65),(1,70),
                (2,10),(2,15),(2,20),(2,25),(2,30),(2,35),(2,40),(2,45),(2,50),(2,55),(2,60),(2,65),(2,70),
                (3,10),(3,15),(3,20),(3,25),(3,30),(3,35),(3,40),(3,45),(3,50),(3,55),(3,60),(3,65),(3,70),
                (4,10),(4,15),(4,20),(4,25),(4,30),(4,35),(4,40),(4,45),(4,50),(4,55),(4,60),(4,65),(4,70),
                (5,10),(5,15),(5,20),(5,25),(5,30),(5,35),(5,40),(5,45),(5,50),(5,55),(5,60),(5,65),(5,70),
                (6,10),(6,15),(6,20),(6,25),(6,30),(6,35),(6,40),(6,45),(6,50),(6,55),(6,60),(6,65),(6,70)]
    
    networks = np.array(networks)
    mask = np.array([False, True, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, False, True, True, True, True, True, False, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, False, False, True, True, True, True, True, True, True, True, True, True, True, False, False, False, True, True, True, True, True, True, True, True, True, True])
    network_sizes = networks[mask]

    classes = []
    for depth, width in network_sizes:
        name = f"{depth}x{width}"
        cls = make_registered_mlp(name, width, depth)
        classes.append(cls)

    return classes

register_all_networks()
