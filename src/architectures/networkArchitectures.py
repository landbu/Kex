import torch
import torch.nn as nn
from typing import Type

networkRegistry: dict[str, Type[nn.Module]] = {}
def registerNetwork(NetworkClass: Type[nn.Module]):
    name = NetworkClass.__name__
    networkRegistry[name] = NetworkClass
    return NetworkClass

########## Dense neural networks ##########

##### Varying number of layers #####

# 28 x 28 -> 20 -> 20 -> 10
@registerNetwork
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
@registerNetwork
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

# 28 x 28 -> 20 -> 20 -> 20 -> 20 -> 10
@registerNetwork
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

# 28 x 28 -> 20 -> 20 -> 20 -> 20 -> 10
@registerNetwork
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

# 28 x 28 -> 50 -> 50 -> 50 -> 10
@registerNetwork
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
@registerNetwork
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
@registerNetwork
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

@registerNetwork
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