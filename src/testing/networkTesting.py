import torch
import torch.nn as nn
from torchvision import datasets

from training.networkTraining import transform

# Load the dataset for testing
testingSet = datasets.MNIST(root = "./data", train = False, transform = transform, download = True)
testingLoader = torch.utils.data.DataLoader(dataset = testingSet, batch_size = 16, shuffle = False)

# Test the accuracy of the model
def testModel(model: nn.Module):
    model.eval()

    numbers = range(0, 10)
    correctClassifications = {number: 0 for number in numbers}
    totalClassifications = {number: 0 for number in numbers}

    with torch.no_grad():
        for images, labels in testingLoader:
            outputs = model(images)
            _, classifications = torch.max(outputs, 1)
            for classification, label in zip(classifications, labels):
                if classification == label:
                    correctClassifications[numbers[label]] += 1
                totalClassifications[numbers[label]] += 1

    for number, correctCount in correctClassifications.items():
        accuracy = 100 * float(correctCount) / totalClassifications[number]
        print(f"Accuracy for number: {number} is {accuracy:.1f} %")

    return correctClassifications, totalClassifications