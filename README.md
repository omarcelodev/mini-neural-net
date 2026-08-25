# Mini Neural Network

A two-layer neural network implemented from scratch using only NumPy — without Keras, PyTorch, TensorFlow, or scikit-learn.

The goal of this project is to understand how neural networks work beneath high-level machine learning frameworks by manually implementing forward propagation, backpropagation, gradient calculation, and weight updates.

The network is trained on the XOR problem as a simple starting point.

## Technologies

* `Python`
* `NumPy` — linear algebra, gradients, matrix operations, and broadcasting
* `Matplotlib` — visualization of the training loss curve

No machine learning frameworks are used.

## Project Structure

```text
mini-neural-net/
├── activations.py   # Sigmoid activation function and derivative
├── loss.py          # Mean Squared Error and derivative
├── network.py       # NeuralNetwork implementation
├── train.py         # Training loop
└── plot.py          # Loss curve visualization
```

## Architecture

The network uses a simple fully connected architecture:

```text
Input Layer (2)
      |
      v
Hidden Layer (8 neurons)
      |
      v
Output Layer (1)
```

### Forward Propagation

```text
Z1 = X · W1 + b1
A1 = sigmoid(Z1)

Z2 = A1 · W2 + b2
A2 = sigmoid(Z2)
```

`A2` represents the network prediction.

### Backpropagation

The gradients are calculated manually using the chain rule:

```text
∂L/∂ŷ
   |
   v
∂L/∂Z
   |
   v
∂L/∂W and ∂L/∂b
```

The weights and biases are then updated using gradient descent:

```text
W -= learning_rate * ∂L/∂W
b -= learning_rate * ∂L/∂b
```

## XOR Dataset

XOR is a classic example of a problem that is not linearly separable, meaning that a simple linear model cannot solve it correctly.

The network therefore uses a hidden layer to learn the nonlinear relationship between the inputs and outputs.

| Input    | Expected Output |
| -------- | --------------: |
| `[0, 0]` |             `0` |
| `[0, 1]` |             `1` |
| `[1, 0]` |             `1` |
| `[1, 1]` |             `0` |

## Training

The example configuration uses:

```python
net = NeuralNetwork(2, 8, 1)

epochs = 10000
learning_rate = 0.3
```

The network progressively adjusts its weights and biases by minimizing the loss through backpropagation and gradient descent.

## Results

Example final predictions:

```text
Expected: [0,     1,     1,     0]
Output:   [0.018, 0.973, 0.973, 0.0312]
```

The model reaches less than approximately 3% absolute error for each XOR example in this training run.

The loss decreases rapidly during the first part of training and gradually stabilizes closer to zero.

![Loss Curve](loss_curve.png)

## Getting Started

Clone the repository:

```bash
git clone https://github.com/omarcelodev/mini-neural-net.git
cd mini-neural-net
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the training:

```bash
python train.py
```

Generate the loss curve:

```bash
python plot.py
```

## What I Learned

This project was created to explore the mathematical and computational foundations of neural networks without relying on high-level machine learning frameworks.

It covers concepts such as:

* matrix operations;
* activation functions;
* loss functions;
* forward propagation;
* backpropagation;
* gradient descent;
* manual parameter updates;
* nonlinear classification.

Implementing these components directly with NumPy helps make the internal mechanics of neural network training easier to understand.

## License

This project is licensed under the [MIT License](LICENSE).

© 2026 Marcelo Gomes
