import numpy as np

def mse(y_pred, y_true): # Função de perda para medir o erro entre as previsões e os valores reais
    return np.mean((y_pred - y_true) ** 2)

def mse_derivada(y_pred, y_true): # Derivada da função de perda para o processo de backpropagation
    return 2 * (y_pred - y_true) / y_true.size

if __name__ == "__main__": # Teste para verificar o funcionamento correto da função mse
    print(mse(np.array([1.0]), np.array([1.0])))
    print(mse(np.array([1.0]), np.array([0.0])))