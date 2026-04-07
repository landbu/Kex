# Helst så använder vi oss bara av networkArchitectures.py
# Om det visar sig att vi behöver använda en builder så kan vi ändra detta

"""
import torch.nn as nn

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
"""