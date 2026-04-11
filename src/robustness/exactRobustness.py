import numpy as np
import time

from robustness.fastLin import computeTwoSideBounds
from robustness.bigM import bigM

def exactRobustness(weights, biases, x0, pNorm, epsilon0: float, originalClass: int, targetClass: int):
    weightsCopy = weights.copy()
    biasesCopy = biases.copy()
    
    numLayers = len(weightsCopy)

    # Saves code for possible later use
    # if targetClass != None:
    #     lastWeight = weightsCopy[-1]
    #     weightsCopy[-1] = (lastWeight[originalClass, :] - lastWeight[targetClass, :]).reshape(1, -1)
    #     lastBias = biasesCopy[-1]
    #     biasesCopy[-1] = lastBias[originalClass] - lastBias[targetClass]
    # else:
    #     raise NotImplementedError("Total exact robustness not yet implemented")
    
    # TODO - Implement automatic search for feasible epsilon
    
    startTime = time.perf_counter()

    maxEpsilon = epsilon0
    
    lowerBounds = [None] * (numLayers + 1)
    upperBounds = [None] * (numLayers + 1)

    lowerBounds[0] = np.full(len(x0), 0)
    upperBounds[0] = np.full(len(x0), 1)
    for l in range(1, numLayers + 1):
        lowerBounds_l, upperBounds_l = computeTwoSideBounds(weights = weightsCopy, biases = biasesCopy, x0 = x0, epsilon = maxEpsilon, pNorm = pNorm, LB = lowerBounds, UB = upperBounds, m = l)
        lowerBounds[l] = lowerBounds_l
        upperBounds[l] = upperBounds_l

    epsilon, adversarialImage = bigM(weights = weightsCopy, biases = biasesCopy, x0 = x0, pNorm = pNorm, maxEpsilon = maxEpsilon, originalClass = originalClass, targetClass = targetClass, lowerBounds = lowerBounds, upperBounds = upperBounds)

    stopTime = time.perf_counter()
    searchTime = stopTime - startTime

    return epsilon, adversarialImage, searchTime
