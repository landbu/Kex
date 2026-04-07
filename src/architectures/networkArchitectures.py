import torch
import torch.nn as nn

########## Dense neural networks ##########
    
# Small dense neural network:
# 28 x 28 -> 20 -> 20 -> 10
class SmallDense1(nn.Module):
    def __init__(self):
        super(SmallDense1, self).__init__()
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

# Small dense neural network:
# 28 x 28 -> 20 -> 20 -> 20 -> 10
class SmallDense2(nn.Module):
    def __init__(self):
        super(SmallDense2, self).__init__()
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
    
########## Convolutional neural networks ##########

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