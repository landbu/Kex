import torch
import torch.nn as nn
from torchvision import datasets, transforms

from NetworkManagement.networks import LeNet

# Load the dataset for training
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])
trainingSet = datasets.MNIST(root = "./data", train = True, transform = transform, download = True)
trainingLoader = torch.utils.data.DataLoader(dataset = trainingSet, batch_size = 16, shuffle = True)

# Train the specified model on the dataset
def trainModel(model: nn.Module):
    model.train()

    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

    numEpochs = 5
    for epoch in range(numEpochs):
        for images, labels in trainingLoader:
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = lossFunction(outputs, labels)
            loss.backward()
            optimizer.step()

# Save the model
def saveModel(model: nn.Module, filename: str):
    torch.save(obj = model.state_dict(), f = f"models/{filename}.pth")

if __name__ == "__main__":
    leNet = LeNet()
    trainModel(leNet)
    saveModel(model = leNet, filename = "leNet")