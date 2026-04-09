from architectures.networkArchitectures import networkRegistry
from testing.networkTesting import testNetwork
from utils.loadNetwork import loadNetwork

# Test the networks
if __name__ == "__main__":
    for name in networkRegistry:
        print(f"----- Testing {name} -----")
        network = loadNetwork(name)
        correctClassifications, totalClassifications = testNetwork(network = network)
        for number, correctCount in correctClassifications.items():
            accuracy = 100 * float(correctCount) / totalClassifications[number]
            print(f"Accuracy for number: {number} is {accuracy:.1f} %")