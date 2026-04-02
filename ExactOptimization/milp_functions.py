import gurobipy as gp
from gurobipy import GRB
import numpy as np
from tqdm import tqdm
from ExactOptimization.other_functions import *

def robustness_setup(X, Y, bridge_model, g_model=None, max_d=1.0): # Make it do a copy instead
    """
    Sets up the neccesary constraints and variables for standard robustnesschecking. That is,
    find the smallest perturbation (now in L1-norm) from X that makes the network not classify it as Y. 
        
    X: Input to be disturbed
    Y: Label, in the form of an index, to avoid
    bridge_model: Bridge model
    max_d: Adjust to cap how big the perturbations are allowed to be

    returns adv variables which is really just the input variables

    """

    if g_model is None:
        g_model = bridge_model.gurobi_model
        output_vars = bridge_model.output_vars
        input_vars = bridge_model.input_vars
    else:
        output_vars = get_MVars(g_model,(10,), "output")
        input_vars = get_MVars(g_model,bridge_model.input_layer.input_shape, "input")

    d = g_model.addMVar(bridge_model.input_layer.input_shape, lb=0, ub=max_d, name="dist") # Helper variable to deal with absolute values
    g_model.addConstr(d >= input_vars - X)
    g_model.addConstr(d >= -(input_vars - X))

    # Binary variables: y_adv[i] = 1 if class i is the winner
    y_adv = g_model.addVars(bridge_model.output_layer.output_width, vtype=GRB.BINARY, name="is_bigger") # This should prob be an MVar

    g_model.addConstr(y_adv[Y] == 0)    
    g_model.addConstr(y_adv.sum() >= 1) # Enforces that there must be an output bigger than z[L][Y]

    epsilon = 0.001   # Gurobipy does not support strict > or <
    for i in range(bridge_model.output_layer.output_width):
        if i == Y: continue
        # "If y_adv[i] is 1, then z[L][i] > z[L][Y]. Since there must be at least one y_adv[i]=1, we don't need equivalence here"
        g_model.addGenConstrIndicator(y_adv[i], 1, output_vars[i] >= output_vars[Y] + epsilon)

    g_model.setObjective(d.sum(), GRB.MINIMIZE)


def standard_robustness_test(bridge_model, g_model=None, print_var_count=False):
    """
    Solves the assumed to be already setup robustness optimization problem and reformats the solution a bit

    adv_vars: Really just the network input vars

    """

    if g_model is None:
        g_model = bridge_model.gurobi_model
        output_vars = bridge_model.output_vars
        input_vars = bridge_model.input_vars
    else:
        output_vars = get_MVars(g_model,(10,), "output")
        input_vars = get_MVars(g_model,bridge_model.input_layer.input_shape, "input")

    if print_var_count:
        print("Starting search for optimal solution")
        print("Num variables not including the robustness testing:", g_model.numVars)
        print("Num constraints not including the robustness testing:", g_model.NumConstrs)
        print("Num binary variables not including the robustness testing:", g_model.NumBinVars)
        g_model.update()
        print("Num variables including the robustness testing:", g_model.numVars)
        print("Num constraints including the robustness testing:", g_model.NumConstrs)
        print("Num binary variables including the robustness testing:", g_model.NumBinVars)


    g_model.optimize()
    if g_model.Status == GRB.OPTIMAL: print("Optimal solution found!")
    else:
        print("No solution found")
        return
    obj_value = g_model.ObjVal
    best_adv_input = input_vars.X
    classification = np.argmax(output_vars.X)

    return obj_value, best_adv_input, classification


def big_robustness_test(bridge_model, samples, labels, viszualize=[], max_d=1.0):
    # Retruns dist mean, variance, and max
    original_model = bridge_model.gurobi_model
    dists = np.zeros_like(labels, dtype=np.float32)
    for i in tqdm(range(len(samples))):
        model_copy = original_model.copy()
        robustness_setup(samples[i], labels[i], bridge_model, g_model=model_copy,max_d=max_d)
        dist, adv_input, classification = standard_robustness_test(bridge_model, g_model=model_copy)
        dists[i] = dist
    
    return np.mean(dists), np.var(dists), np.max(dists)


def feature_selection(bridge_model, Y, mode="independent"): # Döp om
    # mode: mean relavtive, second place relative,independent
    # mean relative means the one which maximizes the distances to the mean
    # independent means the one which makes it as big as possible
    # second place relative means the one which maximizes the distance to the second biggest
    input_vars = bridge_model.input_vars
    output_vars = bridge_model.output_vars
    g_model = bridge_model.gurobi_model
    
    if mode == "mean relative":
        g_model.setObjective(output_vars[Y] - output_vars.sum()/bridge_model.output_layer.output_width, GRB.MAXIMIZE)
    elif mode == "independent":
        g_model.setObjective(output_vars[Y], GRB.MAXIMIZE)
    
    elif mode == "second place relative":
        second_biggest = g_model.addVar(name="m2")
        g_model.addGenConstrMax(output_vars[Y], output_vars.tolist())
        others = [output_vars[i] for i in range(bridge_model.output_layer.output_width) if i != Y]
        g_model.addGenConstrMax(second_biggest, others)
        g_model.setObjective(output_vars[Y]-second_biggest, GRB.MAXIMIZE)
    else: print("Invalid mode")

    g_model.optimize()
    if g_model.Status == GRB.OPTIMAL: print("Optimal solution found!")
    else:
        print("No solution found")
        return
    obj_value = g_model.ObjVal
    image = input_vars.X
    return image, obj_value

def generate_multi_model_input(multi_model, Y, mode="independent"):
    input_vars = multi_model.input_vars
    g_model = multi_model.shared_gurobi_model
    objective_expressions = []

    for bridge in multi_model.bridges:
        output_vars = bridge.output_vars
        
        if mode == "mean relative":
            objective_expressions.append(output_vars[Y] - output_vars.sum()/bridge.output_layer.output_width)
        elif mode == "independent":
            objective_expressions.append(output_vars[Y])
        
        elif mode == "second place relative":
            second_biggest = g_model.addVar(name="m2")
            g_model.addGenConstrMax(output_vars[Y], output_vars.tolist())
            others = [output_vars[i] for i in range(bridge.output_layer.output_width) if i != Y]
            g_model.addGenConstrMax(second_biggest, others)
            objective_expressions.append(output_vars[Y]-second_biggest)
        else: print("Invalid mode")

    objective = gp.quicksum(objective_expressions)
    g_model.setObjective(objective, GRB.MAXIMIZE)

    g_model.optimize()
    if g_model.Status == GRB.OPTIMAL: print("Optimal solution found!")
    else:
        print("No solution found")
        return
    obj_value = g_model.ObjVal
    image = input_vars.X
    return image, obj_value    
        

