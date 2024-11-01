import numpy as np

def mse_loss(predicted, target):
    # Reshape target to match predicted's shape
    target = target.reshape(predicted.shape)
    return np.mean((predicted - target) ** 2)

def mse_loss_grad(predicted, target):
    # Reshape target to match predicted's shape for gradient calculation
    target = target.reshape(predicted.shape)
    return 2 * (predicted - target) / predicted.shape[0]


