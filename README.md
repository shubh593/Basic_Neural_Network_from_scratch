# Basic_Neural_Network_from_scratch
Building a **Neural Network from scratch** is the ultimate way to understand the "Black Box" of Deep Learning. This README should focus on the underlying mathematics—Calculus and Linear Algebra—rather than high-level library abstractions.

---

## 🧠 Basic Neural Network from Scratch (Python)

### 📋 Project Overview

This project implements a fully functional **Multi-Layer Perceptron (MLP)** from the ground up using only **NumPy**. By avoiding high-level frameworks like TensorFlow or PyTorch, this implementation exposes the mechanics of how data flows through neurons, how weights are updated, and how a machine actually "learns" from error.

### 🧬 The Mathematical Architecture

The network follows a classic feedforward structure consisting of an Input layer, one or more Hidden layers, and an Output layer.

#### 1. Forward Propagation

Data is passed through the network. For each layer, we calculate a weighted sum of inputs and apply an activation function (like Sigmoid or ReLU).


#### 2. Loss Function

We calculate the error between the predicted output () and the actual target () using **Mean Squared Error (MSE)**.


#### 3. Backward Propagation (The Heart of Learning)

Using the **Chain Rule** from calculus, we calculate the gradient of the loss function with respect to each weight and bias. This tells us how much to change each value to reduce the error.

#### 4. Weight Updates (Gradient Descent)

We update the weights in the opposite direction of the gradient:


---

### 🚀 Key Features

* **Pure NumPy Implementation**: No deep learning libraries used; focuses on matrix multiplication and vectorized operations.
* **Customizable Topology**: Easily change the number of hidden layers and neurons per layer.
* **Activation Functions**: Includes implementations for **Sigmoid** and its derivative.
* **Epoch-based Training**: Watch the loss decrease in real-time as the network iterates through the data.

---

### 🛠️ Tech Stack

* **Language**: Python 3.x
* **Mathematics/Matrix Ops**: `NumPy`
* **Visualization**: `Matplotlib` (to plot the Loss curve)

---

### 📂 Core Components

* `initialize_parameters()`: Sets weights to small random values and biases to zero.
* `sigmoid()`: The activation function to introduce non-linearity.
* `forward_prop()`: Calculates the linear and activated values for each layer.
* `back_prop()`: Computes the partial derivatives (gradients).
* `update_parameters()`: Adjusts weights using the calculated gradients.

---

### 📊 Training Visualization

A successful training run should show the Loss (Error) decreasing over time as the weights converge toward the optimal solution.

pip install numpy 


2. **Run the Network**:
```python
# Example of a 2-node input, 3-node hidden, 1-node output network
nn = NeuralNetwork(layers=[2, 3, 1])
nn.train(X, y, epochs=1000, learning_rate=0.1)

```



---

### 🛡️ What This Project Teaches

* How **Matrix Transposition** is used in backpropagation.
* The role of the **Learning Rate** in preventing "overshooting" the minimum.
* Why **Non-linear activation functions** are necessary to solve complex problems (like XOR).

