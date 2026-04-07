import numpy as np
import time
import torch
from torchvision import datasets

def fastLin(weights, biases, originalClass, targetClass, input, p, epsilon_0: float, tolerance: float = 1e-6, maxItersExponential: int = 20, maxItersBinary: int = 40):
    m = len(weights)
    lastWeight = weights[-1]
    weights[-1] = (lastWeight[originalClass, :] - lastWeight[targetClass, :]).reshape(1, -1)

    # Exponential search for unsafe epsilon
    startTime = time.perf_counter()

    epsilonLow = 0.0
    epsilonHigh = epsilon_0
    numIters = 0
    while numIters < maxItersExponential:
        lowerBounds = [None] * (m + 1)
        upperBounds = [None] * (m + 1)

        lowerBounds[0] = np.full(len(input), -np.inf)
        upperBounds[0] = np.full(len(input), np.inf)
        for k in range(1, m + 1):
            lowerBounds_k, upperBounds_k = computeTwoSideBounds(weights, biases, input, epsilonHigh, p, lowerBounds, upperBounds, k)
            lowerBounds[k] = lowerBounds_k
            upperBounds[k] = upperBounds_k

        if lowerBounds[m] > 0:
            # epsilon is a certified lower bound, increase using exponential search procedure
            epsilonHigh *= 2
        else:
            # epsilon is not a certified lower bound, move on to binary search
            break

        numIters += 1
        if numIters == maxItersExponential:
            print("Stopping exponential search: reached maximum iterations")
            break
    
    stopTime = time.perf_counter()
    exponentialSearchTime = stopTime - startTime

    # Binary search
    startTime = time.perf_counter()

    numIters = 0
    while numIters < maxItersBinary:
        epsilon = (epsilonHigh + epsilonLow) / 2

        lowerBounds = [None] * (m + 1)
        upperBounds = [None] * (m + 1)

        lowerBounds[0] = np.full(len(input), -np.inf)
        upperBounds[0] = np.full(len(input), np.inf)
        for k in range(1, m + 1):
            lowerBounds_k, upperBounds_k = computeTwoSideBounds(weights, biases, input, epsilon, p, lowerBounds, upperBounds, k)
            lowerBounds[k] = lowerBounds_k
            upperBounds[k] = upperBounds_k

        if lowerBounds[m] > 0:
            # epsilon is a certified lower bound, update binary search
            epsilonLow = epsilon
        else:
            # epsilon is not a certified lower bound, update binary search
            epsilonHigh = epsilon
        
        if (epsilonHigh - epsilonLow) < tolerance:
            print("Stopping binary search: enough accuracy achieved")
            break

        numIters += 1
        if numIters == maxItersBinary:
            print("Stopping binary search: reached maximum iterations")
            break

    stopTime = time.perf_counter()
    binarySearchTime = stopTime - startTime

    return epsilonLow, exponentialSearchTime, binarySearchTime


def computeTwoSideBounds(W, b, x_0, epsilon, p, LB, UB, m): # m is current layer
    # Step 1: Creates matrices A^(k), T^(k), H^(k)
    
    # W[0] = W^(1) ... W[m-1] = W^(m)
    # b[0] = b^(1) ... b[m-1] = b^(m)
    
    A = [None] * m      # A^(0) ... A^(m-1)
    D = [None] * m      # D^(0) ... D^(m-1)
    T = [None] * m      # T^(0) ... T^(m-1)
    H = [None] * m      # H^(0) ... T^(m-1)

    alwaysActive = [None] * m
    alwaysActiveIndices = [None] * m
    alwaysInactive = [None] * m

    alwaysInactiveIndices = [None] * m
    unsure = [None] * m
    unsureIndices = [None] * m

    if m == len(W):
        n_m, _ = np.shape(W[m - 1])
    else:
        _, n_m = np.shape(W[m])

    for k in reversed(range(m)):

        _, n_k = np.shape(W[k])     # Might be problems with this

        # Partitions indices
        alwaysActive[k] = LB[k] >= 0
        alwaysInactive[k] = UB[k] <= 0
        unsure[k] = ~(alwaysActive[k] | alwaysInactive[k])

        alwaysActiveIndices[k] = np.flatnonzero(alwaysActive[k])
        alwaysInactiveIndices[k] = np.flatnonzero(alwaysInactive[k])
        unsureIndices[k] = np.flatnonzero(unsure[k])

        # Constructs D^(k)
        if k == 0:
            D[0] = np.eye(n_k)
        else:
            D[k] = np.zeros((n_k, n_k))
            for r in unsureIndices[k]:
                D[k][r, r] = UB[k][r] / (UB[k][r] - LB[k][r])
            for r in alwaysActiveIndices[k]:
                D[k][r, r] = 1

        # Constructs A^(k)
        if k == m - 1:
            A[m - 1] = W[m - 1] @ D[m - 1]
        else:
            A[k] = A[k + 1] @ W[k] @ D[k]

        # Constructs T^(k) and H^(k)
        T[k] = np.zeros((n_k, n_m))
        H[k] = np.zeros((n_k, n_m))
        for r in unsureIndices[k]:
            for j in range(n_m):
                if A[k][j, r] > 0:
                    T[k][r, j] = LB[k][r]
                elif A[k][j, r] < 0:
                    H[k][r, j] = LB[k][r]

    # Step 2: Calculates lower and upper bounds

    muPlus = np.zeros(n_m)
    muMinus = np.zeros(n_m)
    nu = np.zeros(n_m)
    gammaL = np.zeros(n_m)
    gammaU = np.zeros(n_m)

    if p == 1: q = np.inf
    elif p == 2: q = 2
    elif p == np.inf: q = 1

    for j in range(n_m):
        muPlus[j] = 0
        muMinus[j] = 0
        nu[j] = A[0][j, :] @ x_0 + b[m - 1][j]
        for k in range(1, m):
            muPlus[j] -= A[k][j, :] @ T[k][:, j]
            muMinus[j] -= A[k][j, :] @ H[k][:, j]
            nu[j] += A[k][j, :] @ b[k - 1]

        gammaL[j] = muMinus[j] + nu[j] - epsilon * np.linalg.norm(A[0][j, :], q)
        gammaU[j] = muPlus[j] + nu[j] + epsilon * np.linalg.norm(A[0][j, :], q)

    return gammaL, gammaU