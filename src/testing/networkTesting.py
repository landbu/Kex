import torch
import torch.nn as nn
from torchvision import datasets

from training.networkTraining import transform

def testNetwork(network: nn.Module) -> tuple[dict[int, int], dict[int, int]]:
    # Load the dataset for testing
    testingSet = datasets.MNIST(root = "./data", train = False, transform = transform, download = True)
    testingLoader = torch.utils.data.DataLoader(dataset = testingSet, batch_size = 16, shuffle = False)

    # Test the accuracy of the network
    network.eval()

    numbers = range(0, 10)
    correctClassifications = {number: 0 for number in numbers}
    totalClassifications = {number: 0 for number in numbers}

    with torch.no_grad():
        for images, labels in testingLoader:
            outputs = network(images)
            _, classifications = torch.max(outputs, 1)
            for classification, label in zip(classifications, labels):
                if classification == label:
                    correctClassifications[numbers[label]] += 1
                totalClassifications[numbers[label]] += 1

    return correctClassifications, totalClassifications