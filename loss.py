import numpy as np

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def mse_derivada(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.size

if __name__ == "__main__":
    print(mse(np.array([1.0]), np.array([1.0])))
    print(mse(np.array([1.0]), np.array([0.0])))