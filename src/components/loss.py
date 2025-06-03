import numpy as np

def mse_loss(y_pred, y_true):
   return 0.5 * np.mean((y_pred - y_true)**2)