import time
import torch
import torch.nn as nn
from torchvision import datasets, transforms

# Transform used for data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])

def trainNetwork(network: nn.Module, numEpochs: int, device: torch.device) -> list[float]:
    # Load the dataset for training
    trainingSet = datasets.MNIST(root = "./data", train = True, transform = transform, download = True)
    trainingLoader = torch.utils.data.DataLoader(dataset = trainingSet, batch_size = 16, shuffle = True)

    # Train the specified network on the dataset
    network.to(device)
    network.train()

    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(network.parameters(), lr = 0.001)

    durationPerEpoch = []

    for epoch in range(numEpochs):
        startTime = time.perf_counter()

        for images, labels in trainingLoader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            
            outputs = network(images)
            loss = lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()

        stopTime = time.perf_counter()
        duration = stopTime - startTime
        durationPerEpoch.append(duration)

    return durationPerEpoch

# Save the network
def saveNetwork(network: nn.Module, filename: str):
    torch.save(obj = network.state_dict(), f = f"networks/{filename}.pth")