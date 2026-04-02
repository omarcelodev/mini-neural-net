import numpy as np
from activations import sigmoid, sigmoid_derivada
from loss import mse, mse_derivada

class RedeNeural(): # Classe para representar a rede neural, contendo os pesos, bias e os métodos de forward e backward
    def __init__(self, input_size, hidden_size, output_size): # Incialização dos pesos e bias para a camada oculta 
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(input_size, hidden_size) 
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) 
        self.b2 = np.zeros((1, output_size))

    def forward(self, X): # Método para realizar a propagação para frente, calculando as ativações da camada oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.A1 = sigmoid(self.z1)

        self.z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = sigmoid(self.z2)

        return self.A2

    def backward(self, X, y, lr): # Método para realizar a propagação para trás, calculando os gradientes e atualizando os pesos e bias
        dL = mse_derivada(self.A2, y)
        dZ2 = dL * sigmoid_derivada(self.z2)

        dW2 = np.dot(self.A1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True)

        dZ1 = np.dot(dZ2, self.W2.T) * sigmoid_derivada(self.z1)

        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True)

        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

if __name__ == "__main__": # Teste para verificar o funcionamento correto da rede neural
    net = RedeNeural(2, 4, 1)
    X = np.array([
        [0,0], 
        [0,1], 
        [1,0], 
        [1,1]])
    print(net.forward(X))

