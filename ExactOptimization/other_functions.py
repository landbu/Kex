import torch
import torch.nn as nn
from torchvision import datasets, transforms
from ExactOptimization.general_classes import *
from ExactOptimization.milp_functions import *
import matplotlib.pyplot as plt
import numpy as np


def adverserial_visualization(original_image, original_label, adv_img_flat, obj_val, adv_classification):
    """
    Vizualizes the adverserial example in a 1x3 image grid. The first image is the original image, the second
    is the the perturbed image and the third is the perturbation itself.
    """

    if adv_img_flat is not None:
        adv_img = adv_img_flat.reshape(28, 28)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(original_image.squeeze(), cmap='gray')
        axes[0].set_title(f"Original (Label: {original_label})")
        
        axes[1].imshow(adv_img, cmap='gray')
        axes[1].set_title(f"Adversarial Example. (Classified as {adv_classification})")
        
        diff = adv_img - original_image.squeeze().numpy()
        axes[2].imshow(diff, cmap='RdBu', vmin=-1, vmax=1)
        axes[2].set_title(f"Added Noise (L1: {obj_val:.4f})")
        
        plt.show()
    else:
        print("No adversarial example found.")


def get_MVars(g_model, shape, name):    
    # We use a flat list for Gurobi retrieval first
    flat_vars = []
    
    # If the shape is 1D or naming was flattened (like in your OutputLayer)
    if len(shape) == 1:
        for i in range(shape[0]):
            var = g_model.getVarByName(f"{name}[{i}]")
            if var is None:
                raise ValueError(f"Variable {name}[{i}] not found in model.")
            flat_vars.append(var)
    else:
        # For multi-dimensional CNN shapes
        for idx in np.ndindex(shape):
            idx_str = ",".join(map(str, idx))
            var = g_model.getVarByName(f"{name}[{idx_str}]")
            
            # Fallback: Check if the variables were named with flat indices
            if var is None:
                # Calculate flat index: e.g., for (C,H,W), flat_i = c*H*W + h*W + w
                flat_i = np.ravel_multi_index(idx, shape)
                var = g_model.getVarByName(f"{name}[{flat_i}]")
            
            if var is None:
                raise ValueError(f"Variable {name}[{idx_str}] (or flat equivalent) not found.")
            flat_vars.append(var)
    
    # Use fromlist to create the MVar and then reshape it to match your layer
    return gp.MVar.fromlist(flat_vars).reshape(shape)