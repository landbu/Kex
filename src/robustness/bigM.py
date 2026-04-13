import gurobipy as gp
from gurobipy import GRB
import numpy as np

def bigM(weights, biases, x0, pNorm, maxEpsilon: float, originalClass: int, targetClass: int, lowerBounds, upperBounds, OutputFlag = 0):
    customEnv = gp.Env(empty = True)
    customEnv.setParam("OutputFlag", OutputFlag)
    customEnv.start()

    model = gp.Model(name = "ExactModel", env = customEnv)

    numLayers = len(weights)

    # We assume that z^(0), ..., z^(L-1) and x^(1), ..., x^(L), 
    # and use matching indexing, for example z^(l) = z[l] and x^(l) = x[l]
    x = [None] * (numLayers + 1)
    z = [None] * numLayers
    lambdas = [None] * numLayers

    alwaysActive = [None] * numLayers
    alwaysActiveIndices = [None] * numLayers
    alwaysInactive = [None] * numLayers

    alwaysInactiveIndices = [None] * numLayers
    unsure = [None] * numLayers
    unsureIndices = [None] * numLayers

    if weights[0] is not None:
        W = [None] + weights
    else:
        W = weights

    if biases[0] is not None:
        b = [None] + biases
    else:
        b = biases

    # Input layer
    _, n_0 = np.shape(W[1])
    inputShape = (n_0, )
    inputLayer = model.addMVar(shape = inputShape, lb = lowerBounds[0], ub = upperBounds[0], vtype = GRB.CONTINUOUS, name = "Input")
    z[0] = inputLayer

    # Hidden layers
    for l in range(1, numLayers):
        n_l = np.size(b[l])

        layerShape = (n_l, )

        x[l] = model.addMVar(shape = layerShape, lb = lowerBounds[l], ub = upperBounds[l], vtype = GRB.CONTINUOUS)
        z[l] = model.addMVar(shape = layerShape, lb = 0, ub = np.maximum(0, upperBounds[l]), vtype = GRB.CONTINUOUS)
        lambdas[l] = model.addMVar(shape = layerShape, vtype = GRB.BINARY)

    # Output layer
    n_L = np.size(b[numLayers])
    outputShape = (n_L, )
    outputLayer = model.addMVar(shape = outputShape, lb = lowerBounds[numLayers], ub = upperBounds[numLayers], vtype = GRB.CONTINUOUS, name = "Output")
    x[numLayers] = outputLayer

    # Ensures input is in epsilon-ball
    epsilon = model.addVar(lb = 0, ub = maxEpsilon, vtype = GRB.CONTINUOUS)
    
    if pNorm == 1:
        absVal = model.addVars(n_0)
        for j in range(n_0):
            model.addConstr(inputLayer[j] - x0[j] <= absVal[j])
            model.addConstr(- (inputLayer[j] - x0[j]) <= absVal[j])
        model.addConstr(gp.quicksum(absVal[j] for j in range(n_0)) <= epsilon)
    elif pNorm == 2:
        model.addQConstr(gp.quicksum((inputLayer[j] - x0[j]) * (inputLayer[j] - x0[j]) for j in range(n_0)) <= epsilon**2)
    elif pNorm == np.inf:
        for j in range(n_0):
            model.addConstr(inputLayer[j] - x0[j] >= - epsilon)
            model.addConstr(inputLayer[j] - x0[j] <= epsilon)
    else:
        raise NotImplementedError(f"pNorm = {pNorm} not implemented")

    # MILP constraints (big-M formulation)

    # Pre-ReLU constraints
    for l in range(1, numLayers + 1):
        # Pre-relu constraints
        model.addConstr(x[l] == W[l] @ z[l - 1] + b[l])

    # Relu constraints
    for l in range(1, numLayers):

        # Partitions indices
        alwaysActive[l] = lowerBounds[l] >= 0
        alwaysInactive[l] = upperBounds[l] <= 0
        unsure[l] = ~(alwaysActive[l] | alwaysInactive[l])

        alwaysActiveIndices[l] = np.flatnonzero(alwaysActive[l])
        alwaysInactiveIndices[l] = np.flatnonzero(alwaysInactive[l])
        unsureIndices[l] = np.flatnonzero(unsure[l])

        for activeNeuron in alwaysActiveIndices[l]:
            model.addConstr(z[l][activeNeuron] == x[l][activeNeuron])
        
        for inactiveNeuron in alwaysInactiveIndices[l]:
            model.addConstr(z[l][inactiveNeuron] == 0)

        for unsureNeuron in unsureIndices[l]:
            M_minus = lowerBounds[l][unsureNeuron]
            M_plus = upperBounds[l][unsureNeuron]

            model.addConstr(z[l][unsureNeuron] >= x[l][unsureNeuron])
            model.addConstr(z[l][unsureNeuron] <= x[l][unsureNeuron] - M_minus * (1 - lambdas[l][unsureNeuron]))
            model.addConstr(z[l][unsureNeuron] <= M_plus * lambdas[l][unsureNeuron])
        

    # Different classification constraints

    model.addConstr(outputLayer[targetClass] >= outputLayer[originalClass])

    # Objective function
    model.setObjective(expr = epsilon, sense = GRB.MINIMIZE)    # Finds smallest epsilon-ball that contains different classifications
    model.optimize()

    status = model.Status
    if status != GRB.OPTIMAL:
        # Possibly useful for troubleshooting
        # model.computeIIS()
        # model.write("model.ilp")
        # for c in model.getConstrs():
        #     if c.IISConstr:
        #         print(c.ConstrName, c)
        #
        # for v in model.getVars():
        #     if v.IISLB or v.IISUB:
        #         print(v.VarName, v.lb, v.ub)
        return None, None

    # Should return minimal epsilon and the corresponding adversarial image
    return epsilon.X, inputLayer.X