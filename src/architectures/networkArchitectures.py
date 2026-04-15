from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Type

# Current tags
# "dense", for all dense neural networks
# "CNN", for all CNNs
# "constantWidth", for all networks with constant width for each layer
# "20width", for all networks with a constant width of 20
# "3layers", for all networks with three layers
# "60neurons", for all networks with 60 neurons (feasible for big-M)

@dataclass
class NetworkEntry:
    NetworkClass: Type[nn.Module]
    tags: set[str]

networkRegistry: dict[str, NetworkEntry] = {}
def registerNetwork(*tags: str):
    def registerNetwork(NetworkClass: Type[nn.Module]):
        name = NetworkClass.__name__
        networkRegistry[name] = NetworkEntry(NetworkClass = NetworkClass, tags = set(tags))
        return NetworkClass
    return registerNetwork


########## Dense neural networks ##########

##### Constant number of hidden neurons (60) #####

# 28 x 28 -> 60 -> 10
@registerNetwork("dense", "constantWidth", "60neurons")
class Dense1x60(nn.Module):
    def __init__(self):
        super(Dense1x60, self).__init__()
        self.dense1 = nn.Linear(in_features = 1 * 28 * 28, out_features = 60)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features = 60, out_features = 10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1 * 28 * 28)
        x = self.relu1(self.dense1(x))
        x = self.dense2(x)
        return x

# 28 x 28 -> 30 -> 30 -> 10
@registerNetwork("dense", "constantWidth", "60neurons")
class Dense2x30(nn.Module):
    def __init__(self):
        super(Dense2x30, self).__init__()
        self.dense1 = nn.Linear(in_features = 1 * 28 * 28, out_features = 30)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features = 30, out_features = 30)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features = 30, out_features = 10)

    def forward(self, x : torch.Tensor):
        x = x.view(-1, 1 * 28 * 28)
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.dense3(x)
        return x

# 28 x 28 -> 20 -> 20 -> 20 -> 10
@registerNetwork("dense", "constantWidth", "60neurons", "20width", "3layers")
class Dense3x20(nn.Module):
    def __init__(self):
        super(Dense3x20, self).__init__()
        self.dense1 = nn.Linear(in_features = 1 * 28 * 28, out_features = 20)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features = 20, out_features = 20)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features = 20, out_features = 20)
        self.relu3 = nn.ReLU()
        self.dense4 = nn.Linear(in_features = 20, out_features = 10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1 * 28 * 28)
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.relu3(self.dense3(x))
        x = self.dense4(x)
        return x

# 28 x 28 -> 15 -> 15 -> 15 -> 15 -> 10
@registerNetwork("dense", "constantWidth", "60neurons")
class Dense4x15(nn.Module):
    def __init__(self):
        super(Dense4x15, self).__init__()
        self.dense1 = nn.Linear(in_features = 1 * 28 * 28, out_features = 15)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features = 15, out_features = 15)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features = 15, out_features = 15)
        self.relu3 = nn.ReLU()
        self.dense4 = nn.Linear(in_features = 15, out_features = 15)
        self.relu4 = nn.ReLU()
        self.dense5 = nn.Linear(in_features = 15, out_features = 10)

    def forward(self, x : torch.Tensor):
        x = x.view(-1, 1 * 28 * 28)
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.relu3(self.dense3(x))
        x = self.relu4(self.dense4(x))
        x = self.dense5(x)
        return x

# 28 x 28 -> 10 -> 10 -> 10 -> 10 -> 10 -> 10 -> 10
@registerNetwork("dense", "constantWidth", "60neurons")
class Dense6x10(nn.Module):
    def __init__(self):
        super(Dense6x10, self).__init__()
        self.dense1 = nn.Linear(in_features = 1 * 28 * 28, out_features = 10)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features = 10, out_features = 10)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features = 10, out_features = 10)
        self.relu3 = nn.ReLU()
        self.dense4 = nn.Linear(in_features = 10, out_features = 10)
        self.relu4 = nn.ReLU()
        self.dense5 = nn.Linear(in_features = 10, out_features = 10)
        self.relu5 = nn.ReLU()
        self.dense6 = nn.Linear(in_features = 10, out_features = 10)
        self.relu6 = nn.ReLU()
        self.dense7 = nn.Linear(in_features = 10, out_features = 10)

    def forward(self, x : torch.Tensor):
        x = x.view(-1, 1 * 28 * 28)
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.relu3(self.dense3(x))
        x = self.relu4(self.dense4(x))
        x = self.relu5(self.dense5(x))
        x = self.relu6(self.dense6(x))
        x = self.dense7(x)
        return x

##### Varying number of layers #####

# 28 x 28 -> 20 -> 20 -> 10
@registerNetwork("dense", "constantWidth", "20width")
class Dense2x20(nn.Module):
    def __init__(self):
        super(Dense2x20, self).__init__()
        self.dense1 = nn.Linear(in_features = 1 * 28 * 28, out_features = 20)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features = 20, out_features = 20)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features = 20, out_features = 10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1 * 28 * 28)
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.dense3(x)
        return x

# 28 x 28 -> 20 -> 20 -> 20 -> 10
# Already included!

# 28 x 28 -> 20 -> 20 -> 20 -> 20 -> 10
@registerNetwork("dense", "constantWidth", "20width")
class Dense4x20(nn.Module):
    def __init__(self):
        super(Dense4x20, self).__init__()
        self.dense1 = nn.Linear(in_features = 1 * 28 * 28, out_features = 20)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features = 20, out_features = 20)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features = 20, out_features = 20)
        self.relu3 = nn.ReLU()
        self.dense4 = nn.Linear(in_features = 20, out_features = 20)
        self.relu4 = nn.ReLU()
        self.dense5 = nn.Linear(in_features = 20, out_features = 10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1 * 28 * 28)
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.relu3(self.dense3(x))
        x = self.relu4(self.dense4(x))
        x = self.dense5(x)
        return x

# 28 x 28 -> 20 -> 20 -> 20 -> 20 -> 20 -> 10 
@registerNetwork("dense", "constantWidth", "20width")
class Dense5x20(nn.Module):
    def __init__(self):
        super(Dense5x20, self).__init__()
        self.dense1 = nn.Linear(in_features = 1 * 28 * 28, out_features = 20)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features = 20, out_features = 20)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features = 20, out_features = 20)
        self.relu3 = nn.ReLU()
        self.dense4 = nn.Linear(in_features = 20, out_features = 20)
        self.relu4 = nn.ReLU()
        self.dense5 = nn.Linear(in_features = 20, out_features = 20)
        self.relu5 = nn.ReLU()
        self.dense6 = nn.Linear(in_features = 20, out_features = 10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1 * 28 * 28)
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.relu3(self.dense3(x))
        x = self.relu4(self.dense4(x))
        x = self.relu5(self.dense5(x))
        x = self.dense6(x)
        return x

##### Varying widths #####

# 28 x 28 -> 20 -> 20 -> 20 -> 10
# Already included!

# 28 x 28 -> 50 -> 50 -> 50 -> 10
@registerNetwork("dense", "constantWidth", "3layers")
class Dense3x50(nn.Module):
    def __init__(self):
        super(Dense3x50, self).__init__()
        self.dense1 = nn.Linear(in_features = 1 * 28 * 28, out_features = 50)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features = 50, out_features = 50)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features = 50, out_features = 50)
        self.relu3 = nn.ReLU()
        self.dense4 = nn.Linear(in_features = 50, out_features = 10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1 * 28 * 28)
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.relu3(self.dense3(x))
        x = self.dense4(x)
        return x
    
# 28 x 28 -> 200 -> 200 -> 200 -> 10
@registerNetwork("dense", "constantWidth", "3layers")
class Dense3x200(nn.Module):
    def __init__(self):
        super(Dense3x200, self).__init__()
        self.dense1 = nn.Linear(in_features = 1 * 28 * 28, out_features = 200)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features = 200, out_features = 200)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features = 200, out_features = 200)
        self.relu3 = nn.ReLU()
        self.dense4 = nn.Linear(in_features = 200, out_features = 10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1 * 28 * 28)
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.relu3(self.dense3(x))
        x = self.dense4(x)
        return x
    
# 28 x 28 -> 1000 -> 1000 -> 1000 -> 10
@registerNetwork("dense", "constantWidth", "3layers")
class Dense3x1000(nn.Module):
    def __init__(self):
        super(Dense3x1000, self).__init__()
        self.dense1 = nn.Linear(in_features = 1 * 28 * 28, out_features = 1000)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(in_features = 1000, out_features = 1000)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(in_features = 1000, out_features = 1000)
        self.relu3 = nn.ReLU()
        self.dense4 = nn.Linear(in_features = 1000, out_features = 10)

    def forward(self, x: torch.Tensor):
        x = x.view(-1, 1 * 28 * 28)
        x = self.relu1(self.dense1(x))
        x = self.relu2(self.dense2(x))
        x = self.relu3(self.dense3(x))
        x = self.dense4(x)
        return x

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