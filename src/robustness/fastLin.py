import numpy as np
import time

def fastLin(weights, biases, x0, pNorm, epsilon0: float, originalClass: int, targetClasses: list[int], tolerance: float = 1e-6, maxItersExponential: int = 20, maxItersBinary: int = 40):
    if isinstance(targetClasses, int):
        targetClasses = [targetClasses]
    
    # Copies parameters to avoid modifying them permanently
    weightsCopy = weights.copy()
    biasesCopy = biases.copy()
    
    m = len(weightsCopy)

    lastWeight = weightsCopy[-1]
    weightsCopy[-1] = lastWeight[originalClass, :] - lastWeight[targetClasses, :]
    lastBias = biasesCopy[-1]
    biasesCopy[-1] = np.array([lastBias[originalClass] - lastBias[targetClasses]]).reshape(-1)

    # Exponential search for unsafe epsilon
    startTime = time.perf_counter()

    epsilonLow = 0.0
    epsilonHigh = epsilon0
    numIters = 0
    while numIters < maxItersExponential:
        lowerBounds = [None] * (m + 1)
        upperBounds = [None] * (m + 1)

        lowerBounds[0] = np.full(len(x0), 0)
        upperBounds[0] = np.full(len(x0), 1)
        for k in range(1, m + 1):
            lowerBounds_k, upperBounds_k = computeTwoSideBounds(weightsCopy, biasesCopy, x0, epsilonHigh, pNorm, lowerBounds, upperBounds, k)
            lowerBounds[k] = lowerBounds_k
            upperBounds[k] = upperBounds_k

        if np.any(lowerBounds[m] > 0):
            # epsilon is a certified lower bound, increase using exponential search procedure
            epsilonHigh *= 2
        else:
            # epsilon is not a certified lower bound, move on to binary search
            break

        numIters += 1
        if numIters == maxItersExponential:
            print("Stopping exponential search for Fast-Lin: reached maximum iterations")
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

        lowerBounds[0] = np.full(len(x0), 0)
        upperBounds[0] = np.full(len(x0), 1)
        for k in range(1, m + 1):
            lowerBounds_k, upperBounds_k = computeTwoSideBounds(weightsCopy, biasesCopy, x0, epsilon, pNorm, lowerBounds, upperBounds, k)
            lowerBounds[k] = lowerBounds_k
            upperBounds[k] = upperBounds_k

        if np.all(lowerBounds[m] > 0):
            # epsilon is a certified lower bound, update binary search
            epsilonLow = epsilon
        else:
            # epsilon is not a certified lower bound, update binary search
            epsilonHigh = epsilon
        
        if (epsilonHigh - epsilonLow) < tolerance:
            break

        numIters += 1
        if numIters == maxItersBinary:
            print("Stopping binary search for Fast-Lin: reached maximum iterations")
            break

    stopTime = time.perf_counter()
    binarySearchTime = stopTime - startTime

    return epsilonLow, exponentialSearchTime, binarySearchTime


def computeTwoSideBounds(weights, biases, x0, epsilon: float, pNorm, LB, UB, m: int): # m is the current layer we calculate bounds for
    # Step 1: Create matrices A^(k), T^(k), H^(k)
    
    # Transform weights and biases to satisfy
    # W[1] = W^(1) ... W[m] = W^(m)
    # b[1] = b^(1) ... b[m] = b^(m)
    
    if weights[0] is not None:
        W = [None] + weights
    else:
        W = weights

    if biases[0] is not None:
        b = [None] + biases
    else:
        b = biases

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

    n_m = np.size(b[m])

    for k in reversed(range(m)):
        _, n_k = np.shape(W[k + 1])

        # Partition indices
        alwaysActive[k] = LB[k] >= 0
        alwaysInactive[k] = UB[k] <= 0
        unsure[k] = ~(alwaysActive[k] | alwaysInactive[k])

        alwaysActiveIndices[k] = np.flatnonzero(alwaysActive[k])
        alwaysInactiveIndices[k] = np.flatnonzero(alwaysInactive[k])
        unsureIndices[k] = np.flatnonzero(unsure[k])

        # Construct D^(k)
        if k == 0:
            D[0] = np.eye(n_k)
        else:
            D[k] = np.zeros((n_k, n_k))
            for r in unsureIndices[k]:
                D[k][r, r] = UB[k][r] / (UB[k][r] - LB[k][r])
            for r in alwaysActiveIndices[k]:
                D[k][r, r] = 1

        # Construct A^(k)
        if k == m - 1:
            A[m - 1] = W[m] @ D[m - 1]
        else:
            A[k] = A[k + 1] @ W[k + 1] @ D[k]

        # Construct T^(k) and H^(k)
        T[k] = np.zeros((n_k, n_m))
        H[k] = np.zeros((n_k, n_m))
        for r in unsureIndices[k]:
            for j in range(n_m):
                if A[k][j, r] > 0:
                    T[k][r, j] = LB[k][r]
                elif A[k][j, r] < 0:
                    H[k][r, j] = LB[k][r]

    # Step 2: Calculate lower and upper bounds

    muPlus = np.zeros(n_m)
    muMinus = np.zeros(n_m)
    nu = np.zeros(n_m)
    gammaL = np.zeros(n_m)
    gammaU = np.zeros(n_m)

    if pNorm == 1:
        qNorm = np.inf
    elif pNorm == 2:
        qNorm = 2
    elif pNorm == np.inf:
        qNorm = 1
    else:
        raise NotImplementedError(f"pNorm = {pNorm} not implemented")

    for j in range(n_m):
        nu[j] = A[0][j, :] @ x0 + b[m][j]
        for k in range(1, m):
            muPlus[j] -= A[k][j, :] @ T[k][:, j]
            muMinus[j] -= A[k][j, :] @ H[k][:, j]
            nu[j] += A[k][j, :] @ b[k]

        gammaL[j] = muMinus[j] + nu[j] - epsilon * np.linalg.norm(A[0][j, :], qNorm)
        gammaU[j] = muPlus[j] + nu[j] + epsilon * np.linalg.norm(A[0][j, :], qNorm)

    # TODO - Save relevant matrices to use for next function call

    return gammaL, gammaU