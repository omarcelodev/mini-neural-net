import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivada(z):
    s = sigmoid(z)
    return s * (1 - s)

if __name__ == "__main__":
    print(sigmoid(0))
    print(sigmoid(np.array([-1, 0, 1])))