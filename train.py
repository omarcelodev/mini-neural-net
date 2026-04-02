import numpy as np
from network import RedeNeural
from loss import mse

X = np.array([ # Amostra de dados para o problema XOR
    [0,0],
    [0,1],
    [1,0],
    [1,1]])

y = np.array([[0], [1], [1], [0]]) # Saídas esperadas para o problema XOR

net = RedeNeural(2, 8, 1)
epochs = 10000
lr = 0.3

for epoch in range(epochs): # Loop de treinamento para realizar a propagação para frente, calcular a perda e realizar a propagação para trás
    y_pred = net.forward(X)
    loss = mse(y_pred, y)
    net.backward(X, y, lr)
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

print(net.forward(X)) # Imprime as previsões finais da rede neural 