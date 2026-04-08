import torch

from architectures.networkArchitectures import networkRegistry
from testing.networkTesting import testNetwork

# Test the networks
if __name__ == "__main__":
    for name, NetworkClass in networkRegistry.items():
        print(f"----- Testing {name} -----")
        network = NetworkClass()
        network.load_state_dict(state_dict = torch.load(f"networks/{name}.pth"))
        correctClassifications, totalClassifications = testNetwork(network = network)
        for number, correctCount in correctClassifications.items():
            accuracy = 100 * float(correctCount) / totalClassifications[number]
            print(f"Accuracy for number: {number} is {accuracy:.1f} %")