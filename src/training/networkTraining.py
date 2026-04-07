import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from architectures import LeNet, SmallDense1, SmallDense2

# Transform used for data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

def trainModel(model: nn.Module, numEpochs: int, device: torch.device):
    # Load the dataset for training
    trainingSet = datasets.MNIST(root = "./data", train = True, transform = transform, download = True)
    trainingLoader = torch.utils.data.DataLoader(dataset = trainingSet, batch_size = 16, shuffle = True)

    # Train the specified model on the dataset
    model.to(device)
    model.train()

    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    durationPerEpoch = []

    for epoch in range(numEpochs):
        startTime = time.perf_counter()

        for images, labels in trainingLoader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            outputs = model(images)
            loss = lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()

        stopTime = time.perf_counter()
        duration = stopTime - startTime
        durationPerEpoch.append(duration)

    return durationPerEpoch

# Save the model
def saveModel(model: nn.Module, filename: str):
    torch.save(obj = model.state_dict(), f = f"networks/{filename}.pth")