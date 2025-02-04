# JAXNet: A Lightweight Neural Network Library

JAXNet is a flexible and powerful neural network library built on top of JAX, Equinox, and Optax, designed for rapid prototyping and research.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/mrhashemi/Neural_Net_JAX_Optax_Keras.git
cd Neural_Net_JAX_Optax_Keras

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install dependencies
pip install jax jaxlib equinox optax numpy matplotlib
```

## 🔧 Dependencies

- JAX
- Equinox
- Optax
- NumPy
- Matplotlib

You can install the required libraries using:

```bash
pip install jax jaxlib equinox optax numpy matplotlib
```

> **Note**: For GPU support, follow the official JAX installation guide for your specific system.

## 🚀 Quick Start

### Regression Example

```python
import jax
import jax.numpy as jnp
from jaxnet import JAXNet

# Generate synthetic data
key = jax.random.PRNGKey(0)
x = jax.random.uniform(key, (200, 1)) * 2 * jnp.pi
y = jnp.sin(x) + jax.random.normal(key, x.shape) * 0.1

# Create and train neural network
net = JAXNet(
    architecture=[1, 32, 16, 1],  # Input, hidden layers, output
    learning_rate=0.01,
    activation="relu",
    optimizer="adam"
)

# Train the model
net.fit(x, y, epochs=1000)

# Visualize results
net.visualize_training()
net.visualize_predictions(x, y)

# Make predictions
predictions = net.predict(x)
```

### Classification Example

```python
import jax
import jax.numpy as jnp
from jaxnet import JAXNet
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Preprocess data
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = jnp.array(X)
y = jax.nn.one_hot(jnp.array(y), 3)  # One-hot encode for multi-class

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create neural network for classification
net = JAXNet(
    architecture=[4, 16, 8, 3],  # 4 input features, 3 output classes
    learning_rate=0.01,
    activation="relu",
    optimizer="adam"
)

# Train the model
net.fit(X_train, y_train, epochs=1000, validation_data=(X_test, y_test))

# Visualize training metrics
net.visualize_training()
```

## 🛠 Key Features

- Flexible neural network architecture
- Multiple activation functions
- Support for SGD and Adam optimizers
- Comprehensive metrics tracking
- Visualization utilities
- Model persistence (save/load)
- JIT compilation for high performance

## 📊 Configurable Parameters

- `architecture`: Layer sizes (input, hidden, output)
- `learning_rate`: Optimization step size
- `activation`: Activation function ("sigmoid", "relu", "tanh")
- `optimizer`: Optimization algorithm
- `random_seed`: Reproducibility
- `track_metrics`: Enable/disable metrics tracking

## 📈 Metrics Tracking

JAXNet automatically tracks:
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- R² Score
- Validation metrics

## 💾 Model Persistence

```python
# Save model
net.save_model("model_weights.npy")

# Load model
net.load_model("model_weights.npy")
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🔍 Future Roadmap

- Add more optimizers
- Implement early stopping
- Support for more activation functions
- Enhanced validation metrics
- More visualization options

## 📞 Contact

Project Link: [https://github.com/mrhashemi/jaxnet](https://github.com/mrhashemi/Neural_Net_JAX_Optax_Keras)
