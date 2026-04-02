import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from NetworkManagement.networks import *

device = "cuda" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([
    transforms.ToTensor(),
    #transforms.Normalize((0.1307,), (0.3081,))
])


train_loader = DataLoader(
    datasets.MNIST(root='./Dataset', train=True, download=True, transform=transform),
    batch_size=64, shuffle=True
)

def train_and_save(model, name):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(3):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), f'PyTorchModels/{name}.pth')
    print(f"Model {name} saved.")

# Train both

if __name__=="__main__":
    mlp = get_mlp_modelA()
    train_and_save(mlp, "model_A")
