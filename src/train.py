import torch

from architectures.networkArchitectures import *
from training.networkTraining import trainModel, saveModel

if __name__ == "__main__":
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(device)
    epochs = 3

    print("Training leNet")
    leNet = LeNet()
    trainingTimes = trainModel(model = leNet, numEpochs = epochs)
    for epoch, trainingTime in enumerate(trainingTimes, 1):
        print(f"Training time for epoch {epoch}: {trainingTime:.2f} s.", end = " ")
    print()
    saveModel(model = leNet, filename = "leNet")

    print("Training smallDense1")
    smallDense1 = SmallDense1()
    trainingTimes = trainModel(model = smallDense1, numEpochs = epochs)
    for epoch, trainingTime in enumerate(trainingTimes, 1):
        print(f"Training time for epoch {epoch}: {trainingTime:.2f} s.", end = " ")
    print()
    saveModel(model = smallDense1, filename = "smallDense1")

    print("Training smallDense2")
    smallDense2 = SmallDense2()
    trainingTimes = trainModel(model = smallDense2, numEpochs = epochs)
    for epoch, trainingTime in enumerate(trainingTimes, 1):
        print(f"Training time for epoch {epoch}: {trainingTime:.2f} s.", end = " ")
    print()
    saveModel(model = smallDense2, filename = "smallDense2")