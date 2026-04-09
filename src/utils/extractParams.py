import math
import numpy as np
import torch.nn as nn

# Extracts the matrices W and vectors b for analysis
# Inputs and outputs are assumed to be ordered in ascending order by
# 1: channel, 2: height, 3: width

def extractParams(network: nn.Module, inputShape: tuple[int, int, int]):
    parameters = []             # weights and biases of the network, elements are (W^(n), b^(n))
    outputShape = inputShape    # output from input layer (perhaps unintuitive notation)
    W = None
    b = None
    for layer in network.children():
        W_currentLayer = None
        b_currentLayer = None

        if isinstance(layer, nn.Linear):
            W_currentLayer, b_currentLayer = linearToMatrixForm(layer)
        elif isinstance(layer, nn.Conv2d):
            inputShape = outputShape
            W_currentLayer, b_currentLayer, outputShape = convToMatrixForm(layer, inputShape)
        elif isinstance(layer, nn.AvgPool2d):
            inputShape = outputShape
            W_currentLayer, b_currentLayer, outputShape = poolToMatrixForm(layer, inputShape)
        elif isinstance(layer, nn.ReLU):
            if W is not None and b is not None:
                parameters.append((W, b))
                W = None
                b = None
        else:
            raise ValueError(f"Uncompatible layer: {layer}")
        
        if W is None or b is None:
            W = W_currentLayer
            b = b_currentLayer
        else:
            W = W_currentLayer @ W
            b = W_currentLayer @ b + b_currentLayer

    # Saves final transformation if network does not end with ReLU
    if W is not None and b is not None:
        parameters.append((W, b))

    return parameters

def linearToMatrixForm(layer: nn.Linear):
    W = layer.weight.detach().numpy()
    b = layer.bias.detach().unsqueeze(1).numpy()

    return W, b

def convToMatrixForm(layer: nn.Conv2d, inputShape: tuple[int, int, int]):
    # Getting the parameters of the layer
    _, inputHeight, inputWidth = inputShape

    C_in = layer.in_channels
    C_out = layer.out_channels

    if isinstance(layer.kernel_size, int):
        kernelHeight = kernelWidth = layer.kernel_size
    else: 
        kernelHeight, kernelWidth = layer.kernel_size

    if isinstance(layer.stride, int):
        strideHeight = strideWidth = layer.stride
    else:
        strideHeight, strideWidth = layer.stride

    if isinstance(layer.padding, int):
        paddingHeight = paddingWidth = layer.padding
    else:
        paddingHeight, paddingWidth = layer.padding

    if isinstance(layer.dilation, int):
        dilationHeight = dilationWidth = layer.dilation
    else:
        dilationHeight, dilationWidth = layer.dilation
    
    outputHeight = math.floor((inputHeight + 2 * paddingHeight - dilationHeight * (kernelHeight - 1) - 1) / strideHeight + 1)
    outputWidth = math.floor((inputWidth + 2 * paddingWidth - dilationWidth * (kernelWidth - 1) - 1) / strideWidth + 1)
    outputShape = (C_out, outputHeight, outputWidth)

    # Setting up matrix W and vector b
    W = np.zeros((C_out * outputHeight * outputWidth, C_in * inputHeight * inputWidth))
    b = np.zeros((C_out * outputHeight * outputWidth, 1))
    weight = layer.weight.detach()
    bias = layer.bias.detach()

    # Gets matrix elements by looping over kernel matrix for each output
    for c_out in range(C_out):
        for h_out in range(outputHeight):
            for w_out in range(outputWidth):
                row = c_out * outputHeight * outputWidth + h_out * outputWidth + w_out
                for c_in in range(C_in):
                    for k_h in range(kernelHeight):
                        for k_w in range(kernelWidth):
                            
                            h_in = h_out * strideHeight - paddingHeight + k_h * dilationHeight
                            w_in = w_out * strideWidth - paddingWidth + k_w * dilationWidth

                            if 0 <= h_in < inputHeight and 0 <= w_in < inputWidth:
                                col = c_in * inputHeight * inputWidth + h_in * inputWidth + w_in
                                W[row, col] = weight[c_out][c_in][k_h][k_w]

    # Gets vector elements corresponding to the output channel by looping over each output
    for c_out in range(C_out):
        for h_out in range(outputHeight):
            for w_out in range(outputWidth):
                row = c_out * outputHeight * outputWidth + h_out * outputWidth + w_out
                b[row] = bias[c_out]

    return W, b, outputShape

def poolToMatrixForm(layer: nn.AvgPool2d, inputShape: tuple[int, int, int]):
    # Getting the parameters of the layer
    C, inputHeight, inputWidth = inputShape

    if isinstance(layer.kernel_size, int):
        kernelHeight = kernelWidth = layer.kernel_size
    else: 
        kernelHeight, kernelWidth = layer.kernel_size

    if isinstance(layer.stride, int):
        strideHeight = strideWidth = layer.stride
    else:
        strideHeight, strideWidth = layer.stride

    if isinstance(layer.padding, int):
        paddingHeight = paddingWidth = layer.padding
    else:
        paddingHeight, paddingWidth = layer.padding
    
    outputHeight = math.floor((inputHeight + 2 * paddingHeight - kernelHeight) / strideHeight + 1)
    outputWidth = math.floor((inputWidth + 2 * paddingWidth - kernelWidth) / strideWidth + 1)
    outputShape = (C, outputHeight, outputWidth)

    # Setting up matrix W and zero vector b
    W = np.zeros((C * outputHeight * outputWidth, C * inputHeight * inputWidth))
    b = np.zeros((C * outputHeight * outputWidth, 1))
    scale = 1 / (kernelHeight * kernelWidth)

    # Gets matrix elements by looping over kernel matrix for each output
    for c in range(C):
        for h_out in range(outputHeight):
            for w_out in range(outputWidth):
                row = c * outputHeight * outputWidth + h_out * outputWidth + w_out
                for k_h in range(kernelHeight):
                    for k_w in range(kernelWidth):
                            
                        h_in = h_out * strideHeight - paddingHeight + k_h
                        w_in = w_out * strideWidth - paddingWidth + k_w

                        if 0 <= h_in < inputHeight and 0 <= w_in < inputWidth:
                            col = c * inputHeight * inputWidth + h_in * inputWidth + w_in
                            W[row, col] = scale

    return W, b, outputShape