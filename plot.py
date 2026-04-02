import matplotlib.pyplot as plt
import numpy as np
from network import RedeNeural
from loss import mse

# Plot da curva de perda durante o treinamento da rede neural

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])

y = np.array([[0], [1], [1], [0]])

losses = []

net = RedeNeural(2, 8, 1)
epochs = 10000
lr = 0.3

for epoch in range(epochs):
    y_pred = net.forward(X)
    loss = mse(y_pred, y)
    net.backward(X, y, lr)
    if epoch % 100 == 0:
        losses.append(loss)

epochs_axis = list(range(0, epochs, 100)) #
plt.plot(epochs_axis, losses)

plt.title("Curva de perda")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.savefig("loss_curve.png")
plt.show()