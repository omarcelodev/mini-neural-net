import numpy as np
from network import RedeNeural
from loss import mse

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])

y = np.array([[0], [1], [1], [0]])

net = RedeNeural(2, 8, 1)
epochs = 10000
lr = 0.3

for epoch in range(epochs):
    y_pred = net.forward(X)
    loss = mse(y_pred, y)
    net.backward(X, y, lr)
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

print(net.forward(X))