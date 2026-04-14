from src.architectures.networkArchitectures import networkRegistry
from src.testing.networkTesting import testNetwork
from src.utils.loadNetwork import loadNetwork

# Test the networks
if __name__ == "__main__":
    DETAILED_TEST = False

    for name in networkRegistry:
        print(f"----- Testing {name} -----")
        network = loadNetwork(name)
        correctClassifications, totalClassifications = testNetwork(network = network)

        if DETAILED_TEST:
            for number, correctCount in correctClassifications.items():
                accuracy = 100 * float(correctCount) / totalClassifications[number]
                print(f"Accuracy for number: {number} is {accuracy:.1f} %")

        sumOfCorrectClassifications = sum(correctClassifications.values())
        sumOfTotalClassifications = sum(totalClassifications.values())
        accuracy = 100 * float(sumOfCorrectClassifications) / sumOfTotalClassifications
        print(f"Total accuracy is {accuracy:.1f} %")