import torch
import torch.nn as nn

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
    
class CNNBuilder:
    def __init__(self, input_shape):
        # input_shape = (C, H, W)
        self.c, self.h, self.w = input_shape
        self.layers = []

    def add_conv(self, out_channels, kernel_size=3, stride=1, padding=0):
        self.layers.append(nn.Conv2d(self.c, out_channels, kernel_size, stride, padding))

        # update spatial dimensions
        self.h = (self.h + 2*padding - kernel_size) // stride + 1
        self.w = (self.w + 2*padding - kernel_size) // stride + 1
        self.c = out_channels

    def add_relu(self):
        self.layers.append(nn.ReLU())

    def add_pool(self, kernel_size=2, stride=2):
        self.layers.append(nn.AvgPool2d(kernel_size, stride))

        self.h = (self.h - kernel_size) // stride + 1
        self.w = (self.w - kernel_size) // stride + 1

    def build(self, num_classes):
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(self.c * self.h * self.w, num_classes))
        return nn.Sequential(*self.layers)

def get_mlp_modelA():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 50),
        nn.ReLU(),
        nn.Linear(50, 30),
        nn.ReLU(),
        nn.Linear(30, 10)
    )

def get_mlp_modelB():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 20),
        nn.ReLU(),
        nn.Linear(20, 30),
        nn.ReLU(),
        nn.Linear(30, 10)
    )

def get_mlp_modelC():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 40),
        nn.ReLU(),
        nn.Linear(40, 20),
        nn.ReLU(),
        nn.Linear(20, 20),
        nn.ReLU(),
        nn.Linear(30, 10)
    )

def get_mlp_modelD():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 20),
        nn.ReLU(),
        nn.Linear(50, 10),
    )

def get_cnn_model():
    builder = CNNBuilder((1, 28, 28))

    builder.add_conv(6, kernel_size=4)
    builder.add_pool(kernel_size=3)
    builder.add_relu()

    builder.add_conv(2, kernel_size=3, stride=2)
    builder.add_pool(kernel_size=3)
    builder.add_relu()


    return builder.build(num_classes=10)