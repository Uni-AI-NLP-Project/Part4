import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import copy

epsilon = 1e-15
debug = 0


class FCLayer:
    def __init__(self, input_size, output_size, activation):
        """
        Args:
            input_size (int): Input shape of the layer
            output_size (int): Output of the layer
            activation (str): activation function
        """
        self.input = None
        self.output = None

        # Initialize weights and biases (weights are initialized using HE-Initialization)
        self.weights = torch.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = torch.zeros((1, output_size))
        self.activation = activation

        # Define variables and hyperparameters used in Adam optimization
        self.m_weights = torch.zeros((input_size, output_size))
        self.v_weights = torch.zeros((input_size, output_size))
        self.m_biases = torch.zeros((1, output_size))
        self.v_biases = torch.zeros((1, output_size))
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8

    # def forward(self, x):
    #     """
    #     Forward pass
    #
    #     Args:
    #         x (Tensor): Numerical values of the data
    #     """
    #     self.input = x
    #     output_no_activation = torch.dot(self.input, self.weights) + self.biases
    #
    #     if self.activation == "relu":
    #         self.output = torch.maximum(torch.zeros(), output_no_activation)
    #         self.output = torch.maximum(torch.zeros_like(output_no_activation), output_no_activation)
    #
    #     elif self.activation == "softmax":
    #         exp_values = np.exp(output_no_activation - np.max(output_no_activation, axis=-1, keepdims=True))
    #         self.output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    #
    #     else:
    #         raise Exception(f"Given activation function '{self.activation}' is invalid.")
    #
    #     return self.output

    def forward(self, x):
        self.input = x
        output_no_activation = torch.matmul(self.input, self.weights) + self.biases

        if self.activation == "relu":
            self.output = torch.maximum(torch.tensor(0.0), output_no_activation)
        elif self.activation == "softmax":
            exp_values = torch.exp(output_no_activation - torch.max(output_no_activation, dim=-1, keepdim=True).values)
            self.output = exp_values / torch.sum(exp_values, dim=-1, keepdim=True)
        else:
            raise Exception(f"Given activation function '{self.activation}' is invalid.")

        return self.output

    def backward(self, d_values, learning_rate, t):
        """
        Backpropagation

        Args:
            d_values (float): Derivative of the output
            learning_rate (float): Learning rate for gradient descent
            t (int): Timestep
        """
        # Calculate the derivative of the ReLU function
        if self.activation == "relu":
            d_values = d_values * (self.output > 0)

        # Get the derivative of the softmax Function
        elif self.activation == "softmax":
            for i, gradient in enumerate(d_values):
                if len(gradient.shape) == 1:  # For single instance
                    gradient = gradient.reshape(-1, 1)
                jacobian_matrix = np.diagflat(gradient) - np.dot(gradient, gradient.T)
                d_values[i] = np.dot(jacobian_matrix, self.output[i])

        # Calculate the derivative with respect to the weight and bias (one with weight and one with bias)
        d_weights = np.dot(self.input.T, d_values)
        d_biases = np.sum(d_values, axis=0, keepdims=True)
        # Limit the derivative to avoid really big or small numbers
        d_weights = np.clip(d_weights, -1.0, 1.0)
        d_biases = np.clip(d_biases, -1.0, 1.0)

        # Calculate the gradient with respect to the input
        d_inputs = np.dot(d_values, self.weights.T)

        # Update the weights and biases using the learning rate and their derivatives
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases

        # Update weights using m and v values
        m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * d_weights
        v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (d_weights ** 2)
        m_hat_weights = m_weights / (1 - self.beta1 ** t)
        v_hat_weights = v_weights / (1 - self.beta2 ** t)
        self.weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)

        # Update biases using m and v values
        m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * d_biases
        v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (d_biases ** 2)
        m_hat_biases = m_biases / (1 - self.beta1 ** t)
        v_hat_biases = v_biases / (1 - self.beta2 ** t)
        self.biases -= learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)

        return d_inputs


class CreateModel:
    def __init__(self, input_size, output_size, hidden_size):
        """
        Creating the model using layers (written from scratch)
        Args:
            input_size (int): Input shape of the data
            output_size (int): Output size of the layer
            hidden_size (list): size of hidden layers
        """
        self.layer1 = FCLayer(input_size=input_size, output_size=hidden_size[0], activation="relu")
        self.layer2 = FCLayer(input_size=hidden_size[0], output_size=hidden_size[1], activation="relu")
        self.layer3 = FCLayer(input_size=hidden_size[1], output_size=output_size, activation="softmax")

    def forward(self, inputs):
        """
        Forward propagation

        Args:
            x (Tensor): A tensor consist of neumerical values of the data
        """
        # Calculate the output
        output1 = self.layer1.forward(inputs)
        output2 = self.layer2.forward(output1)
        output3 = self.layer3.forward(output2)

        return output3

    def train(self, inputs, targets, n_epochs, initial_learning_rate, decay, plot_training_results=False):
        """
        This function does the training process of the model,
            First forward propagation is done, then the loss and accuracy are calculated,
            After that the backpropagation is done.

        Args:
            inputs (Tensor): _description_
            targets (Tensor): _description_
            initial_learning_rate (float): _description_
            decay (float): _description_
            plot_training_results (bool, optional): _description_. Defaults to False.
        """
        # Define timestep
        t = 0

        # Define lists for loss and accuracy
        loss_log = []
        accuracy_log = []

        for epoch in range(n_epochs):
            output = self.forward(inputs=inputs)  # calculate the forward pass
            loss = calculate_CCE(output, targets)  # calculate loss
            predicted_labels = torch.argmax(output, dim=1)
            comparison = (predicted_labels == targets).float()
            accuracy = torch.mean(comparison)  # calculate the accuracy

            # backward
            output_grad = 6 * (output - targets) / output.shape[0]
            t += 1
            learning_rate = initial_learning_rate / (1 + decay * epoch)
            grad_3 = self.layer3.backward(output_grad, learning_rate, t)
            grad_2 = self.layer2.backward(grad_3, learning_rate, t)
            grad_1 = self.layer1.backward(grad_2, learning_rate, t)

            # Add the loss and accuracy to the list
            if plot_training_results:
                loss_log.append(loss)
                accuracy_log.append(accuracy)

            # print training results
            print(f"Epoch {epoch} // Loss: {loss} // Accuracy: {accuracy}")

        # Draw plot if needed
        if plot_training_results:
            plt.plot(range(n_epochs), loss_log, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.show()

            plt.plot(range(n_epochs), accuracy_log, label='Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training Accuracy Curve')
            plt.legend()
            plt.show()


def load_data(path_to_csv):
    data = pd.read_csv(path_to_csv)

    # Separate features (X) and target (y)
    X = data.drop('Diabetes_binary', axis=1)  # Replace 'target_column' with the name of your target column
    y = data['Diabetes_binary']

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
    )

    if debug > 0:
        print(f"X_train shape: {X_train.shape}, type: {type(X_train)}")
        print("X_val shape:", X_val.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_val shape:", y_val.shape)
        print("y_test shape:", y_test.shape)

    X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.int)
    y_val = torch.tensor(y_val.to_numpy(), dtype=torch.int)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.int)

    return X_train, X_val, X_test, y_train, y_val, y_test


def calculate_CCE(predictions, targets) -> float:
    """
    Calculate the loss (Categorical (binary) Cross Entropy):
    1. Filter out irrelevant prediction-entries
    2. Change range of probabilty from [0, 1] to [epsilon, 1] to avoid log(0) calculations
    3. Do log_e(entry) for each entry
    4. Average them (and multiply by -1)
    *  Original formula: -sum_over_i{sum_over_j[y_i,j * log_e(prediction_i,j)]}
    *  Machine learning formula: -avg{sum_over_j[y_i,j * log_e(prediction_i,j)]}
    *  Fomula is simplified to just -avg{[log_e(prediction_i,(y_i))]} for clasifiction.
    :param predictions:
    :param targets:
    :return:
    """

    relevant_predictions = predictions[torch.arange(len(targets)), targets]
    predictions_no_zeros = torch.clip(relevant_predictions, min=epsilon, max=1)
    predictions_log_e = torch.log(predictions_no_zeros)
    loss = -torch.mean(predictions_log_e)

    if debug >= 0:
        print(f"predictions\n{predictions}")
        print(f"relevant_predictions\n{relevant_predictions}")
        print(f"predictions_log_e\n{predictions_log_e}")
        print(f"loss\n{loss}")

    return loss


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    if debug > 0:
        print(f"Using {device} device")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data('../diabetes_binary_health_indicators_BRFSS2015.csv')

    # Define hyperparameters for training
    INPUT_SIZE = 21
    HIDDEN_SIZE = [64, 32]
    OUTPUT_SIZE = 2

    print("TRAINING STARTED")

    # Create the Neural Network model
    nn = CreateModel(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE)
    nn.train(X_train, y_train, initial_learning_rate=0.001, decay=0.001, n_epochs=100, plot_training_results=True)
