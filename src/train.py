import torch

from architectures.networkArchitectures import networkRegistry
from training.networkTraining import trainNetwork, saveNetwork

# Train the networks
if __name__ == "__main__":
    currentDevice = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    epochs = 3
    for name, NetworkClass in networkRegistry.items():
        print(f"Training {name}")
        network = NetworkClass()
        trainingTimes = trainNetwork(network = network, numEpochs = epochs, device = currentDevice)
        for epoch, trainingTime in enumerate(trainingTimes, 1):
            print(f"Training time for epoch {epoch}: {trainingTime:.2f} s")
        saveNetwork(network = network, filename = name)
        print(f"Saved {name}")