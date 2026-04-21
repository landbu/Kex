import torch
import torch.nn as nn
import gurobipy as gp
from gurobipy import GRB
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm

class ModelBridge():
    def __init__(self,torch_model, gurobi_model=None):
        customEnv = gp.Env(empty = True)
        OutputFlag = 1
        customEnv.setParam("OutputFlag", OutputFlag)  
        customEnv.start()
        if gurobi_model is None:
            self.gurobi_model =  gp.Model("LargeModel", env = customEnv)
        else: self.gurobi_model = gurobi_model
        self.torch_model = torch_model
        self.layers = []
        self.gurobi_model.setParam("TimeLimit", 90)  # seconds

    def add_layer(self,layer):
        self.layers.append(layer)

    def save():
        # Saves the model, probably pickled, implement later
        pass

    def add_input_layer(self, input_size, L0=0, U0=1):
        self.input_layer = InputLayer(self.gurobi_model,input_size, L0, U0)
        self.input_vars = self.input_layer.network_input_vars

    def setup_gurobi_logic(self):
        # Note: self.input_vars and self.output_vars are gurobipy dicts, but the hidden variables are
        # passed around inside numpy arrays
        self.gurobi_model.update()
        self.gurobi_model.terminate() # Potential magic
        self.gurobi_model.Params.OutputFlag = 0
        z = self.input_vars
        for i, layer in enumerate(self.layers):
            z = layer.setup_gurobi_forward_logic(z)
            self.gurobi_model.update()
            #print(f"Just initialized layer {i}, of type {type(self.layers[i])} lower bounds for that layer are of shape {z.lb.shape} and look like:\n {z.lb}\n The upper bounds are of shape {z.ub.shape} and look like: {z.ub}")

        self.output_layer = OutputLayer(self)
        self.output_vars = self.output_layer.setup_gurobi_forward_logic(z)

    def load_layers_from_pytorch_model(self, L0=0, U0=1, input_shape=None, create_input_layer=True):
            """
            Initializes ModelBridge layers by inspecting self.torch_model.
            Works for both the old nn.Sequential paradigm and the new custom Module paradigm.
            """
            self.layers = [] # Reset layers list if re-loading
            
            # 1. Handle Input Shape Inference
            if input_shape is None:
                # Look through children until we find a layer with dimensions to infer shape
                for m in self.torch_model.children():
                    if isinstance(m, nn.Linear):
                        input_shape = (m.in_features,)
                        break
                    elif isinstance(m, nn.Conv2d):
                        raise ValueError("For Convolutional models, please provide input_shape as (C, H, W)")
                    elif isinstance(m, nn.Flatten):
                        continue 
            
            # Initialize the input layer in Gurobi
            if create_input_layer: 
                self.add_input_layer(input_shape, L0, U0)
            current_shape = input_shape

            # 2. Iterate through children of self.torch_model
            # .children() works for Sequential containers and classes using setattr() for layers.
            for m in self.torch_model.children():
                # Skip Softmax as it is handled by the robustness logic/MILP objectives
                if isinstance(m, (nn.Softmax, nn.LogSoftmax)):
                    continue

                if isinstance(m, nn.Linear):
                    # Paradigms like GeneratedMLP flatten in forward(), not as a module.
                    # If current shape is multi-dimensional (e.g., 28x28), we auto-flatten here.
                    if len(current_shape) > 1:
                        flat_size = int(np.prod(current_shape))
                        if m.in_features == flat_size:
                            self.add_layer(Reshape(self, (flat_size,)))
                            current_shape = (flat_size,)

                    weights = m.weight.detach().cpu().numpy()
                    bias = m.bias.detach().cpu().numpy()
                    
                    new_layer = Linear(self, weights, bias)
                    self.add_layer(new_layer)
                    current_shape = (m.out_features,)

                elif isinstance(m, nn.ReLU):
                    self.add_layer(ReluLayer(self))

                elif isinstance(m, nn.Conv2d):
                    kernel = m.weight.detach().cpu().numpy()
                    bias = m.bias.detach().cpu().numpy()
                    stride = m.stride
                    
                    new_layer = Convolution(self, kernel, bias, stride, bound_tightening="ia")
                    self.add_layer(new_layer)
                    
                    # Update Shape Inference for next layers
                    out_channels, _, k_h, k_w = kernel.shape
                    h_out = (current_shape[1] - k_h) // stride[0] + 1
                    w_out = (current_shape[2] - k_w) // stride[1] + 1
                    current_shape = (out_channels, h_out, w_out)

                elif isinstance(m, nn.AvgPool2d):
                    kernel_size = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size)
                    stride = m.stride if isinstance(m.stride, tuple) else (m.stride, m.stride)
                    
                    self.add_layer(MeanPooling(self, kernel_size, stride))
                    
                    h_out = (current_shape[1] - kernel_size[0]) // stride[0] + 1
                    w_out = (current_shape[2] - kernel_size[1]) // stride[1] + 1
                    current_shape = (current_shape[0], h_out, w_out)

                elif isinstance(m, nn.Flatten):
                    flat_size = int(np.prod(current_shape))
                    self.add_layer(Reshape(self, (flat_size,)))
                    current_shape = (flat_size,)

                else:
                    # If the module is a nested Sequential, we should process its children recursively
                    if isinstance(m, nn.Sequential):
                        # For simple MLPs, we can just treat the internal layers as part of the main chain
                        for sub_m in m.children():
                            # Repeat logic for sub_m or recursively call a helper
                            pass
                    else:
                        print(f"Warning: Layer type {type(m)} not recognized by bridge.")



class MultiModel():
    def __init__(self, input_shape, L0=0, U0=1):
        self.shared_gurobi_model = gp.Model()
        self.bridges = []
        self.input_layer = InputLayer(self.shared_gurobi_model, input_shape, L0, U0)
        self.input_vars = self.input_layer.network_input_vars
        self.output_layers = []
        self.shared_gurobi_model.setParam("TimeLimit", 90)  # seconds


    def add_networks(self,pytorch_models):
        for pytorch_model in pytorch_models:
            bridge = ModelBridge(pytorch_model, gurobi_model=self.shared_gurobi_model)
            bridge.input_layer = self.input_layer
            bridge.input_vars = self.input_vars
            bridge.load_layers_from_pytorch_model(create_input_layer=False)
            self.bridges.append(bridge)
            bridge.setup_gurobi_logic()
            self.output_layers.append(bridge.output_layer)

    

class Layer:
    def __init__(self, g_model):
        self.g_model = g_model
        self.fall_back = 500.0

    #def __init__(self, model_bridge):
    #    self.model_bridge = model_bridge
    #    self.g_model = model_bridge.gurobi_model
    #    self.fall_back = 500.0

    def exact_bound_tightening(self, var, lb=None,ub=None):
        if lb is None:
            self.g_model.setObjective(var, GRB.MINIMIZE)
            self.g_model.update()
            self.g_model.optimize()
            if self.g_model.SolCount > 0: var.lb = self.g_model.ObjVal
            elif self.g_model.Status == GRB.TIME_LIMIT: var.lb = self.g_model.ObjBound
            else: var.lb = -self.fall_back 

        if ub is None:
            self.g_model.setObjective(var, GRB.MAXIMIZE)
            self.g_model.update()
            self.g_model.optimize()
            if self.g_model.SolCount > 0: var.ub = self.g_model.ObjVal
            elif self.g_model.Status == GRB.TIME_LIMIT: var.ub = self.g_model.ObjBound
            else: var.ub = self.fall_back


    def fall_back_bound_tightening(self, var, lb=None,ub=None):
        if not lb is None:
            var.lb = -self.fall_back
        if not ub is None:
            var.ub = self.fall_back

    
    def setup_gurobi_forward_logic():
        # This must be implemented for each layer
        raise NotImplementedError


class Linear(Layer):
    def __init__(self,model_bridge, weights, bias):
        super().__init__(model_bridge.gurobi_model)
        self.weights = weights
        self.biases = bias
        self.width = weights.shape[0]
        self.prev_width = weights.shape[1]
        self.fall_back = 50000.0
        self.setup_gurobi_forward_logic = self.setup_gurobi_forward_logic_ia

        # The adjustability of the bound tightening is pretty bad right now

    def do_exact_bound_tightening(self):
        self.setup_gurobi_forward_logic = self.setup_gurobi_forward_logic_exact

    def set_ia_bounds(self, layer_input_vars):
        W_neg = np.where(self.weights > 0, 0, self.weights)
        W_pos = np.where(self.weights < 0, 0, self.weights)
        L = layer_input_vars.lb
        U = layer_input_vars.ub

        lb = W_pos @ L + W_neg @ U + self.biases
        ub = W_pos @ U + W_neg @ L + self.biases
        self.output_vars.lb = lb
        self.output_vars.ub = ub

    def setup_gurobi_forward_logic_ia(self, layer_input_vars):
        self.output_vars = self.g_model.addMVar((self.width))
        forward_expession = self.weights @ layer_input_vars + self.biases
        self.g_model.addConstr(self.output_vars == forward_expession)
        self.set_ia_bounds(layer_input_vars)
        return self.output_vars

    def setup_gurobi_forward_logic_exact(self,layer_input_vars):
        n_prev = self.prev_width
        z = self.g_model.addMVar((self.width), lb=-GRB.INFINITY, ub=GRB.INFINITY)

        pbar = tqdm(range(self.width))
        for i in pbar:
            pbar.set_description(f"Setting up neuron {i+1}")
            expression_i = gp.quicksum(self.weights[i][j] * layer_input_vars[j] for j in range(n_prev)) + self.biases[i]            
            self.g_model.addConstr(z[i] == expression_i)
            self.exact_bound_tightening(z[i])

        self.output_vars = z
        return self.output_vars


class InputLayer():
    def __init__(self, g_model, input_shape, L0, U0):
        self.g_model = g_model
        self.network_input_vars = self.g_model.addMVar(input_shape, lb=L0,ub=U0,name="input")
        self.input_shape = input_shape

    def setup_gurobi_forward_logic(self):
        return self.network_input_vars


class OutputLayer(Layer):
    def __init__(self, model_bridge):
        super().__init__(model_bridge.gurobi_model)

    def setup_gurobi_forward_logic(self, layer_input_vars):
        #self.network_output_vars = self.g_model.addMVar(layer_input_vars.shape, name="output")
        #self.g_model.addConstr(layer_input_vars==self.network_output_vars)

        self.network_output_vars = layer_input_vars
        for idx in np.ndindex(layer_input_vars.shape):
            idx_str = ",".join(map(str, idx))
            # This ensures a 3D tensor is named 'output[0,0,0]' instead of 'output[0]'
            layer_input_vars[idx].VarName = f"output[{idx_str}]"
        #for i, var in enumerate(self.network_output_vars.tolist()):
        #    var.VarName = f"output[{i}]"
        self.g_model.update()
        #self.network_output_vars.VarName = "output"
        self.output_width  = layer_input_vars.shape[0]
        return self.network_output_vars


class ReluLayer(Layer):
    def __init__(self, model_bridge):
        super().__init__(model_bridge.gurobi_model)
        
    def setup_gurobi_forward_logic(self, layer_input_vars):
        # Rename things temporarily for clarity
        x = layer_input_vars
        L = x.lb
        U = x.ub
        model = self.g_model

        # Initialize output variables
        z = self.g_model.addMVar(layer_input_vars.shape)
        z.ub = np.maximum(0, layer_input_vars.ub) 
        z.lb = np.maximum(0, layer_input_vars.lb)

        # Using switches for variables we know we won't need is unnecesary.
        # The logic bellow ensures we don't do that.
        always_active = L >= 0
        never_active = U <= 0
        ambiguous = ~(always_active | never_active)
        model.addConstr(z[always_active] == x[always_active])
        model.addConstr(z[never_active] == 0)
        n_ambiguous = np.sum(ambiguous)
        lambd = model.addMVar(n_ambiguous, vtype=GRB.BINARY)

        x_flat = x.reshape(-1)
        z_flat = z.reshape(-1)
        L_flat = L.reshape(-1)
        U_flat = U.reshape(-1)
        mask_flat = ambiguous.reshape(-1)
        x_amb = x_flat[mask_flat]
        z_amb = z_flat[mask_flat]
        L_amb = L_flat[mask_flat]
        U_amb = U_flat[mask_flat]

        # Set up switch logic for only the variables which need it
        model.addConstr(z_amb >= x_amb)
        model.addConstr(z_amb >= 0)
        model.addConstr(z_amb <= x_amb - L_amb * (1 - lambd))
        model.addConstr(z_amb <= U_amb * lambd)
        self.output_vars = z
        return self.output_vars


class MeanPooling(Layer):
    def __init__(self, model_bridge, pool_size, stride, bound_tightening="ia"):
        super().__init__(model_bridge.gurobi_model)
        self.h_stride, self.w_stride = stride
        self.pool_size = pool_size
        self.bound_tightening = bound_tightening


    def calc_ia_bounds(self, layer_input_vars):
        # extract bounds from previous layer
        l_raw = np.array([v.lb for v in layer_input_vars.reshape(-1)]).reshape(layer_input_vars.shape)
        u_raw = np.array([v.ub for v in layer_input_vars.reshape(-1)]).reshape(layer_input_vars.shape)

        # Convert to Tensors (Batch=1, Depth=D, H, W)
        L = torch.from_numpy(l_raw).float().unsqueeze(0)
        U = torch.from_numpy(u_raw).float().unsqueeze(0)

        kernel_size = self.pool_size
        stride = (self.h_stride, self.w_stride)

        with torch.no_grad():
            lower = F.avg_pool2d(L, kernel_size=kernel_size, stride=stride)
            upper = F.avg_pool2d(U, kernel_size=kernel_size, stride=stride)

        lower_bounds = lower.squeeze(0).numpy()
        upper_bounds = upper.squeeze(0).numpy()

        self.output_vars.lb = lower_bounds
        self.output_vars.ub = upper_bounds

    def setup_gurobi_forward_logic(self, layer_input_vars):
        D, H, W = layer_input_vars.shape
        K_h, K_w = self.pool_size
        H_out = (H - K_h) // self.h_stride + 1
        W_out = (W - K_w) // self.w_stride + 1
        window_size = K_h*K_w
        self.output_vars = self.g_model.addMVar((D,H_out,W_out))

        if self.bound_tightening == "ia":
            self.calc_ia_bounds(layer_input_vars)

        pbar = tqdm(total=D * H_out * W_out) if self.bound_tightening == "exact" else None

        for d in range(D):
            for hh in range(H_out):
                for ww in range(W_out):
                    # Calc position of top left corner
                    top_left_x, top_left_y = ww*self.w_stride, hh*self.h_stride
                    image_slice = layer_input_vars[d, top_left_y:top_left_y+K_h, top_left_x:top_left_x+K_w].reshape(-1)
                    sum_expression = gp.quicksum(image_slice[i] for i in range(window_size))
                    self.g_model.addConstr(self.output_vars[d,hh,ww] ==  sum_expression / window_size)
                    if self.bound_tightening == "exact":
                        self.exact_bound_tightening(self.output_vars[d, hh, ww])
                        pbar.update(1)
        if pbar: pbar.close()
        return self.output_vars


class Reshape(Layer):
    def __init__(self, model_bridge, out_shape):
        super().__init__(model_bridge.gurobi_model)
        self.out_shape = out_shape

    def setup_gurobi_forward_logic(self, layer_input_vars):
        return layer_input_vars.reshape(self.out_shape)


class Convolution(Layer):
    def __init__(self, model_bridge, kernel_tensor, biases, stride, bound_tightening):
        super().__init__(model_bridge.gurobi_model)
        self.kernel_tensor = kernel_tensor
        self.biases = biases
        self.h_stride, self.w_stride = stride
        self.bound_tightening = bound_tightening


    def calc_ia_bounds(self, layer_input_vars):
        # extract bounds from previous layer
        l_raw = layer_input_vars.lb
        u_raw = layer_input_vars.ub
        
        # Convert to Tensors (Batch=1, Depth=D, H, W)
        L = torch.from_numpy(l_raw).float().unsqueeze(0)
        U = torch.from_numpy(u_raw).float().unsqueeze(0)
        W = torch.from_numpy(self.kernel_tensor).float() 
        
        W_pos = torch.clamp(W, min=0)
        W_neg = torch.clamp(W, max=0)
        stride = (self.h_stride, self.w_stride)

        with torch.no_grad():
            upper = F.conv2d(U, W_pos, stride=stride) + F.conv2d(L, W_neg, stride=stride)
            lower = F.conv2d(L, W_pos, stride=stride) + F.conv2d(U, W_neg, stride=stride)

        # Final output + Bias
        bias = torch.from_numpy(self.biases).float().view(1, -1, 1, 1)
        upper_bounds = (upper + bias).squeeze(0).numpy()
        lower_bounds = (lower + bias).squeeze(0).numpy()

        self.output_vars.ub = upper_bounds
        self.output_vars.lb = lower_bounds

    def setup_gurobi_forward_logic(self, layer_input_vars):
        # This is valid cross correlation.
        D, H, W = layer_input_vars.shape
        N_f, D, K_h, K_w = self.kernel_tensor.shape # Assume it is of appropriate dimensions
        H_out = (H-K_h)//self.h_stride+1
        W_out = (W-K_w)//self.w_stride+1

        self.output_vars = self.g_model.addMVar((N_f,H_out,W_out), lb=-GRB.INFINITY, ub=GRB.INFINITY)
        if self.bound_tightening == "ia": self.calc_ia_bounds(layer_input_vars)
        number_of_multiplications_per_convolution_step = self.kernel_tensor[0].size
        pbar = tqdm(total=N_f * H_out * W_out) if self.bound_tightening == "exact" else None


        for f in range(N_f):
            current_filter = self.kernel_tensor[f].flatten()
            for hh in range(H_out):
                for ww in range(W_out):
                    # Calc position of top left corner
                    top_left_x, top_left_y = ww*self.w_stride, hh*self.h_stride
                    image_slice = layer_input_vars[:, top_left_y:top_left_y+K_h, top_left_x:top_left_x+K_w].reshape(-1)
                    convolution_expression =  \
                    gp.quicksum(image_slice[i] * current_filter[i] for i in range(number_of_multiplications_per_convolution_step))
                    self.g_model.addConstr(self.output_vars[f, hh, ww] == convolution_expression + self.biases[f])

                    if self.bound_tightening == "exact":
                        self.exact_bound_tightening(self.output_vars[f, hh, ww])
                        pbar.update(1)
                        pbar.set_description(f"Setting up neuron {f+1}/{N_f}, {hh+1}/{H_out}, {ww+1}/{W_out}")

        if pbar: pbar.close()

        return self.output_vars