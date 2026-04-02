import numpy as np

def sigmoid(z): # Função de ativação para mapear os valores para o intervalo (0, 1)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivada(z): # Derivada para o processo de backpropagation
    s = sigmoid(z)
    return s * (1 - s)

if __name__ == "__main__": # Teste para verificar o funcionamento correto da função sigmoid
    print(sigmoid(0))
    print(sigmoid(np.array([-1, 0, 1])))