from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Type


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

    GeneratedMLP.__name__ = f"Dense{depth}x{width}"
    registerNetwork("dense", "constantWidth", name)(GeneratedMLP)

    return GeneratedMLP


def register_all_networks():
    original_network_sizes = [(1,60),(2,30),(3,20),(4,15),(6,10),
                              (2,20),(4,20),(5,20),(3,50),(3,200)]  #,(3,1000)]
    
    v1_bonus_network_sizes = [(2, 41), (3, 38), (2, 12), (2, 28), (2, 23),
                              (2, 22), (3, 31), (2, 34), (3, 42), (3, 51),
                              (3, 25), (2, 29), (3, 41), (3, 19), (2, 39),
                              (3, 47), (2, 37), (3, 33), (2, 18), (3, 29),
                              (3, 43), (3, 32), (2, 20), (3, 12), (2, 40),
                              (2, 19), (3, 36), (2, 32), (3, 48)]
    
    #v2_bonus_network_sizes = [(1,35), (2,25), (3,36), (5,15), (4,20), (5, 23)
    #                          (4,10), (5,25), (6,10), (6,15), (1,30), (2,17), (1, 48)]
    
    network_sizes = original_network_sizes + v1_bonus_network_sizes

    classes = []
    for depth, width in network_sizes:
        name = f"{depth}x{width}"
        cls = make_registered_mlp(name, width, depth)
        classes.append(cls)

    return classes

register_all_networks()

"""
########## Convolutional neural networks ##########

@registerNetwork("CNN")
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, kernel_size = 5, padding = 2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = 5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        self.dense1 = nn.Linear(in_features = 16 * 5 * 5, out_features = 120)
        self.relu3 = nn.ReLU()
        self.dense2 = nn.Linear(in_features = 120, out_features = 84)
        self.relu4 = nn.ReLU()
        self.dense3 = nn.Linear(in_features = 84, out_features = 10)

    def forward(self, x: torch.Tensor):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu3(self.dense1(x))
        x = self.relu4(self.dense2(x))
        x = self.dense3(x)
        return x


"""