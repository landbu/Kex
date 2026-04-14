import numpy as np
import time

from src.robustness.fastLin import computeTwoSideBounds
from src.robustness.fastLin import fastLin
from src.robustness.bigM import bigM

def exactRobustness(weights, biases, x0, pNorm, epsilon0: float, originalClass: int, targetClasses: list[int], maxIters: int = 5):
    if isinstance(targetClasses, int):
        targetClasses = [targetClasses]

    numLayers = len(weights)

    startTime = time.perf_counter()

    epsilons = []
    adversarialImages = []

    # Solves the big-M formulation for each targetClass
    for targetClass in targetClasses:
        certifiedEpsilon, _, _ = fastLin(weights = weights, biases = biases, x0 = x0, pNorm = pNorm, epsilon0 = epsilon0, originalClass = originalClass, targetClasses = targetClasses, tolerance = 0.01)
        maxEpsilon = certifiedEpsilon * 3
        
        # Gradually doubles the upper bound of epsilon until we find the exact epsilon
        numIters = 0
        while numIters < maxIters:
            lowerBounds = [None] * (numLayers + 1)
            upperBounds = [None] * (numLayers + 1)

            lowerBounds[0] = np.full(len(x0), 0)
            upperBounds[0] = np.full(len(x0), 1)
            for l in range(1, numLayers + 1):
                lowerBounds_l, upperBounds_l = computeTwoSideBounds(weights = weights, biases = biases, x0 = x0, epsilon = maxEpsilon, pNorm = pNorm, LB = lowerBounds, UB = upperBounds, m = l)
                lowerBounds[l] = lowerBounds_l
                upperBounds[l] = upperBounds_l

            epsilon, adversarialImage = bigM(weights = weights, biases = biases, x0 = x0, pNorm = pNorm, maxEpsilon = maxEpsilon, originalClass = originalClass, targetClass = targetClass, lowerBounds = lowerBounds, upperBounds = upperBounds)

            if epsilon is not None:
                break
            
            maxEpsilon *= 2
            numIters += 1
            if numIters == maxIters:
                print("Stopping exponential search for exact robustness: reached maximum iterations")
                break

        epsilons.append(epsilon)
        adversarialImages.append(adversarialImages)

    minIndex = np.argmin(epsilons)
    epsilon = epsilons[minIndex]
    adversarialImage = adversarialImages[minIndex]

    stopTime = time.perf_counter()
    searchTime = stopTime - startTime

    return epsilon, adversarialImage, searchTime
