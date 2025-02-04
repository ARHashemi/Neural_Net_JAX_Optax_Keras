"""
JAXNet: A lightweight neural network library built with JAX, Equinox, and Optax.

This library provides a simple yet flexible implementation of feedforward neural networks
using JAX's automatic differentiation and just-in-time compilation capabilities.

Author: M.R. Hashemi; A.R. Hashemi
License: MIT
"""

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable, Dict, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class MetricsTracker:
    """
    Track and store training metrics over time.
    
    Attributes:
        losses: List of training loss values
        metrics: Dictionary of additional metric values (e.g., MSE, MAE, R2)
    """
    losses: List[float]
    metrics: Dict[str, List[float]]
    
    def update(self, loss: float, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Update metrics with new values from training iteration.

        Args:
            loss: Current iteration loss value
            metrics: Dictionary of current metric values
        """
        self.losses.append(float(loss))
        if metrics:
            for key, value in metrics.items():
                self.metrics.setdefault(key, []).append(float(value))

class JAXNet:
    """
    A flexible feedforward neural network implementation using JAX.
    
    Features:
        - Configurable architecture with arbitrary layer sizes
        - Multiple activation functions (sigmoid, ReLU, tanh)
        - Choice of optimizers (SGD, Adam)
        - Comprehensive metrics tracking
        - Training validation
        - Model persistence
        - Visualization utilities
    """
    
    def __init__(
        self,
        architecture: List[int],
        learning_rate: float = 0.01,
        activation: Union[str, Callable] = "sigmoid",
        optimizer: str = "adam",
        random_seed: int = 42,
        track_metrics: bool = True
    ):
        """
        Initialize a new neural network.

        Args:
            architecture: List specifying number of neurons in each layer
                        [input_size, hidden1_size, ..., output_size]
            learning_rate: Step size for gradient updates
            activation: Activation function for hidden layers
                       Either "sigmoid", "relu", "tanh" or a custom function
            optimizer: Optimization algorithm ("sgd" or "adam")
            random_seed: Seed for reproducible weight initialization
            track_metrics: Whether to store training metrics

        Example:
            >>> net = JAXNet([2, 64, 32, 1], learning_rate=0.01, activation="relu")
        """
        self.architecture = architecture
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.track_metrics = track_metrics
        
        # Initialize random key
        self.rng = jax.random.PRNGKey(random_seed)
        
        # Set activation function
        self._set_activation(activation)
        
        # Initialize network parameters
        self.network = self._build_network()
        
        # Configure optimizer
        self._setup_optimizer(optimizer)
        
        # Initialize metrics tracking
        self.metrics = MetricsTracker([], {}) if track_metrics else None

    def _set_activation(self, activation: Union[str, Callable]) -> None:
        """
        Set the network's activation function.

        Args:
            activation: Either a string identifier or a custom function
        """
        self.activation = {
            "sigmoid": jax.nn.sigmoid,
            "relu": jax.nn.relu,
            "tanh": jax.nn.tanh
        }.get(activation, activation) if isinstance(activation, str) else activation

    def _build_network(self) -> eqx.Module:
        """
        Construct the neural network architecture using Equinox.
        
        Returns:
            An Equinox Module representing the neural network
        """
        class FeedForwardNet(eqx.Module):
            """Neural network implementation with configurable architecture."""
            layers: List[eqx.nn.Linear]
            activation: Callable

            def __init__(self, architecture: List[int], key: jax.random.PRNGKey, activation: Callable):
                self.activation = activation
                self.layers = []
                
                for i, (in_size, out_size) in enumerate(zip(architecture[:-1], architecture[1:])):
                    key, subkey = jax.random.split(key)
                    
                    # Use He initialization for ReLU, Xavier for others
                    if activation == jax.nn.relu:
                        init_scale = jnp.sqrt(2.0 / in_size)
                    else:
                        init_scale = jnp.sqrt(1.0 / in_size)
                        
                    self.layers.append(
                        eqx.nn.Linear(
                            in_size, 
                            out_size, 
                            use_bias=True, 
                            key=subkey,
                            kernel_init=lambda key, shape: init_scale * jax.random.normal(key, shape)
                        )
                    )

            def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
                """Forward pass through the network."""
                for layer in self.layers[:-1]:
                    x = self.activation(layer(x))
                return self.layers[-1](x)

        return FeedForwardNet(self.architecture, self.rng, self.activation)

    def _setup_optimizer(self, optimizer_name: str) -> None:
        """
        Configure the optimization algorithm.

        Args:
            optimizer_name: Name of the optimizer to use
        """
        optimizer_map = {
            "sgd": optax.sgd(self.learning_rate),
            "adam": optax.adam(self.learning_rate)
        }
        
        if optimizer_name not in optimizer_map:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}. Choose from {list(optimizer_map.keys())}")
            
        self.optimizer = optimizer_map[optimizer_name]
        self.opt_state = self.optimizer.init(eqx.filter(self.network, eqx.is_array))

    def _compute_performance_metrics(# 
# A simple neural network using JAX with optax optimizer and a custom defined loss function.
#
# This program is licensed under MIT
#
import jax
import jax.numpy as jnp
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Callable, Dict, Union
from dataclasses import dataclass
import numpy as np


@dataclass
class TrainingHistory:
    """
    Stores the history of training, including loss and additional performance metrics.

    Attributes:
        losses (List[float]): Stores loss values over epochs.
        metrics (Dict[str, List[float]]): Tracks additional metrics (MSE, MAE, R²).
    """
    losses: List[float]
    metrics: Dict[str, List[float]]

    def add(self, loss: float, additional_metrics: Optional[Dict[str, float]] = None):
        """
        Append a loss value and optionally additional metrics.

        Args:
            loss (float): Computed loss for the current epoch.
            additional_metrics (Optional[Dict[str, float]]): Dictionary of additional metrics.
        """
        self.losses.append(float(loss))
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.metrics.setdefault(key, []).append(float(value))


class JAXMLPRegressor:
    """
    A multi-layer perceptron (MLP) regressor using JAX, Equinox, and Optax.

    Features:
    - Supports multiple activation functions.
    - Allows different optimizers (SGD, Adam).
    - Tracks training history with metrics like MSE, MAE, and R².
    - Uses cosine similarity and mean absolute error (MAE) as a combined loss function.
    """

    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.01,
        activation: Union[str, Callable] = "sigmoid",
        optimizer_type: str = "sgd",
        random_seed: int = 42,
        track_history: bool = True,
    ):
        """
        Initialize the neural network.

        Args:
            layer_sizes (List[int]): Number of neurons per layer.
            learning_rate (float): Learning rate for the optimizer.
            activation (Union[str, Callable]): Activation function 
                (Options: "sigmoid", "relu", "tanh", "leaky_relu", "elu", "softplus", 
                "gelu", "swish", "selu", "softmax", "log_sigmoid", "log_softmax").
            optimizer_type (str): Optimization algorithm ("sgd" or "adam").
            random_seed (int): Random seed for reproducibility.
            track_history (bool): Whether to store training loss and metrics.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.track_history = track_history
        self.key = jax.random.PRNGKey(random_seed)

        # Set activation function
        self.activation = {
            "sigmoid": jax.nn.sigmoid,
            "relu": jax.nn.relu,
            "tanh": jax.nn.tanh,
            "leaky_relu": jax.nn.leaky_relu,
            "elu": jax.nn.elu,
            "softplus": jax.nn.softplus,
            "gelu": jax.nn.gelu,
            "swish": jax.nn.swish,
            "selu": jax.nn.selu,
            "softmax": jax.nn.softmax,
            "log_sigmoid": jax.nn.log_sigmoid,
            "log_softmax": jax.nn.log_softmax
        }.get(activation, activation) if isinstance(activation, str) else activation

        # Initialize the model
        self.model = self._initialize_model()

        # Set optimizer
        self.optimizer = {
            "sgd": optax.sgd(learning_rate),
            "adam": optax.adam(learning_rate),
        }[optimizer_type]

        self.opt_state = self.optimizer.init(eqx.filter(self.model, eqx.is_array))
        self.history = TrainingHistory([], {}) if track_history else None

    def _initialize_model(self) -> eqx.Module:
        """Creates and returns the MLP model using Equinox."""
        class MLP(eqx.Module):
            layers: List[eqx.nn.Linear]
            activation: Callable

            def __init__(self, layer_sizes: List[int], key: jax.random.PRNGKey, activation: Callable):
                self.activation = activation
                self.layers = []
                
                for i, (input_dim, output_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                    key, subkey = jax.random.split(key)
                    self.layers.append(eqx.nn.Linear(input_dim, output_dim, use_bias=True, key=subkey))

            def __call__(self, x):
                for layer in self.layers[:-1]:
                    x = self.activation(layer(x))
                return self.layers[-1](x)  # No activation on the last layer

        return MLP(self.layer_sizes, self.key, self.activation)

    def _compute_metrics(self, y_true: jnp.ndarray, y_pred: jnp.ndarray) -> Dict[str, jnp.float16]:
        """Compute Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² metrics."""
        mse = jnp.mean((y_true - y_pred) ** 2)
        mae = jnp.mean(jnp.abs(y_true - y_pred))
        r2 = 1 - jnp.sum(jnp.square(y_true - y_pred)) / jnp.sum(jnp.square(y_true - jnp.mean(y_true)))
        return {"mse": jnp.float16(mse), "mae": jnp.float16(mae), "r2": jnp.float16(r2)}

    @eqx.filter_jit
    def _perform_training_step(
        self, model: eqx.Module, opt_state: optax.OptState, x: jnp.ndarray, y: jnp.ndarray
    ) -> Tuple[eqx.Module, optax.OptState, float, Dict[str, float]]:
        """Executes a single training step using cosine similarity + MAE loss."""
        def loss_fn(model):
            y_pred = jax.vmap(model)(x)

            # Cosine Similarity Loss
            dot_product = jnp.sum(y_pred * y, axis=1)
            norm_preds = jnp.linalg.norm(y_pred, axis=1)
            norm_y = jnp.linalg.norm(y, axis=1)
            cosine_similarity = dot_product / (norm_preds * norm_y + 1e-8)
            cosine_loss = 1 - jnp.mean(cosine_similarity)

            # Mean Absolute Error (MAE) Loss
            mae_loss = jnp.mean(jnp.abs(y_pred - y))

            # Combined Loss
            combined_loss = 0.5 * cosine_loss + 0.5 * mae_loss
            return combined_loss, y_pred

        (loss, y_pred), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, opt_state = self.optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        metrics = self._compute_metrics(y, y_pred)
        return model, opt_state, loss, metrics

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """Generates predictions for input data."""
        return jax.vmap(self.model)(x)

    def train(self, x: jnp.ndarray, y: jnp.ndarray, epochs: int = 1000, verbose: bool = True):
        """
        Trains the model.

        Args:
            x (jnp.ndarray): Input training data.
            y (jnp.ndarray): Target labels.
            epochs (int): Number of epochs to train.
            verbose (bool): Whether to print training progress.
        """
        for epoch in range(epochs):
            self.model, self.opt_state, loss, metrics = self._perform_training_step(self.model, self.opt_state, x, y)
            if self.track_history:
                self.history.add(loss, metrics)
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.4f}, MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")
        self,
        y_true: jnp.ndarray,
        y_pred: jnp.ndarray
    ) -> Dict[str, float]:
        """
        Calculate various performance metrics.

        Args:
            y_true: Ground truth values
            y_pred: Model predictions

        Returns:
            Dictionary of computed metrics
        """
        mse = jnp.mean((y_true - y_pred) ** 2)
        mae = jnp.mean(jnp.abs(y_true - y_pred))
        
        # Compute R² score
        ss_res = jnp.sum((y_true - y_pred) ** 2)
        ss_tot = jnp.sum((y_true - jnp.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-10))  # Add epsilon to prevent division by zero
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "r2": float(r2)
        }

    @eqx.filter_jit
    def _training_iteration(
        self,
        network: eqx.Module,
        opt_state: optax.OptState,
        x_batch: jnp.ndarray,
        y_batch: jnp.ndarray
    ) -> Tuple[eqx.Module, optax.OptState, float, Dict[str, float]]:
        """
        Perform one training iteration.

        Args:
            network: Current network state
            opt_state: Current optimizer state
            x_batch: Input batch
            y_batch: Target batch

        Returns:
            Updated network, optimizer state, loss, and metrics
        """
        def loss_fn(network: eqx.Module) -> Tuple[float, jnp.ndarray]:
            predictions = jax.vmap(network)(x_batch)
            loss = jnp.mean((predictions - y_batch) ** 2)
            return loss, predictions

        (loss, predictions), gradients = eqx.filter_value_and_grad(loss_fn, has_aux=True)(network)
        updates, opt_state = self.optimizer.update(gradients, opt_state, network)
        network = eqx.apply_updates(network, updates)
        metrics = self._compute_performance_metrics(y_batch, predictions)
        
        return network, opt_state, loss, metrics

    def fit(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        epochs: int = 1000,
        validation_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
        verbose: bool = True,
        verbose_interval: int = 100
    ) -> Optional[MetricsTracker]:
        """
        Train the neural network.

        Args:
            x_train: Training input data
            y_train: Training target data
            epochs: Number of training epochs
            validation_data: Optional tuple of (x_val, y_val) for validation
            verbose: Whether to print training progress
            verbose_interval: Number of epochs between progress updates

        Returns:
            MetricsTracker if track_metrics=False, else None

        Example:
            >>> x_train = jnp.array([[0.0], [1.0]])
            >>> y_train = jnp.array([[0.0], [1.0]])
            >>> net.fit(x_train, y_train, epochs=1000)
        """
        metrics_tracker = MetricsTracker([], {}) if not self.track_metrics else self.metrics
        
        for epoch in range(epochs):
            # Training step
            self.network, self.opt_state, loss, metrics = self._training_iteration(
                self.network,
                self.opt_state,
                x_train,
                y_train
            )
            
            # Track metrics
            if metrics_tracker is not None:
                metrics_tracker.update(loss, metrics)
            
            # Compute validation metrics
            if validation_data is not None:
                x_val, y_val = validation_data
                val_predictions = self.predict(x_val)
                val_metrics = self._compute_performance_metrics(y_val, val_predictions)
                val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}
                
                if metrics_tracker is not None:
                    metrics_tracker.update(loss, val_metrics)
            
            # Progress reporting
            if verbose and epoch % verbose_interval == 0:
                metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
                print(f"Epoch {epoch}/{epochs} - loss: {loss:.4f} - {metrics_str}")
        
        return metrics_tracker if not self.track_metrics else None

    def predict(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Generate predictions for input data.

        Args:
            x: Input data array

        Returns:
            Model predictions

        Example:
            >>> predictions = net.predict(jnp.array([[0.5]]))
        """
        return jax.vmap(self.network)(x)

    def visualize_training(
        self,
        metrics: Optional[List[str]] = None,
        use_log_scale: bool = True,
        figure_size: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Visualize training metrics over time.

        Args:
            metrics: List of metrics to plot (default: all available metrics)
            use_log_scale: Whether to use log scale for loss plot
            figure_size: Size of the figure (width, height)

        Raises:
            ValueError: If metrics tracking is disabled
        """
        if self.metrics is None:
            raise ValueError("Metrics tracking is disabled. Initialize with track_metrics=True")
            
        if metrics is None:
            metrics = list(self.metrics.metrics.keys())
            
        n_plots = len(metrics) + 1
        fig, axes = plt.subplots(1, n_plots, figsize=figure_size)
        if n_plots == 1:
            axes = [axes]
            
        # Plot loss history
        axes[0].plot(self.metrics.losses)
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Epoch')
        if use_log_scale:
            axes[0].set_yscale('log')
            
        # Plot additional metrics
        for i, metric in enumerate(metrics, 1):
            if metric in self.metrics.metrics:
                axes[i].plot(self.metrics.metrics[metric])
                axes[i].set_title(metric.replace('_', ' ').title())
                axes[i].set_xlabel('Epoch')
                
        plt.tight_layout()
        plt.show()

    def visualize_predictions(
        self,
        x: jnp.ndarray,
        y: jnp.ndarray,
        title: str = "Model Predictions vs Actual Data",
        figure_size: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Visualize model predictions against actual data.

        Args:
            x: Input data
            y: True target values
            title: Plot title
            figure_size: Size of the figure (width, height)
        """
        plt.figure(figsize=figure_size)
        predictions = self.predict(x)
        
        plt.scatter(x, y, label="Ground Truth", alpha=0.5)
        plt.scatter(x, predictions, label="Predictions", alpha=0.5)
        plt.title(title)
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, filepath: str) -> None:
        """
        Save model parameters to a file.

        Args:
            filepath: Path to save the model weights
        """
        weights = eqx.filter(self.network, eqx.is_array)
        jnp.save(filepath, weights)

    def load_model(self, filepath: str) -> None:
        """
        Load model parameters from a file.

        Args:
            filepath: Path to the saved model weights
        """
        weights = jnp.load(filepath)
        self.network = eqx.combine(weights, eqx.filter(self.network, eqx.is_inexact))
