import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
import tqdm
import copy

debug = 0


class NeuralNetwork(nn.Module):
    def __init__(self, in_features: int = 21, hidden_layers_sizes: list = None, out_features: int = 2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(21, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
        # self.input_layer = nn.Linear(in_features, hidden_layers_sizes[0])
        # self.hidden_layers = []
        # for i in range(1, len(hidden_layers_sizes)):
        #     self.hidden_layers.append(nn.Linear(hidden_layers_sizes[i - 1], hidden_layers_sizes[i]))

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class NeuralNetwork2(nn.Module):
    def __init__(self, inSize, hiddenSizes, outSize):
        super().__init__()
        self.inSize = inSize
        self.hiddenSize = hiddenSizes
        self.outSize = outSize

        self.lin1 = nn.Linear(inSize, hiddenSizes[0])
        self.lin2 = nn.Linear(hiddenSizes[0], hiddenSizes[1])
        self.lin3 = nn.Linear(hiddenSizes[1], outSize)

        self.bn1 = nn.BatchNorm2d(hiddenSizes[0])
        self.bn2 = nn.BatchNorm2d(hiddenSizes[1])

    def forward(self, x):
        x = nn.ReLU(self.bn1(self.lin1(x)))
        x = nn.ReLU(self.bn2(self.lin2(x)))
        x = self.lin3(x)
        return x


class NeuralNetwork3(nn.Module):
    def __init__(self):
        super(NeuralNetwork3, self).__init__()

        # Define the layers
        self.fc1 = nn.Linear(21, 64)  # Input layer to 1st hidden layer
        self.relu1 = nn.ReLU()  # Activation function for 1st hidden layer
        self.fc2 = nn.Linear(64, 32)  # 1st hidden layer to 2nd hidden layer
        self.relu2 = nn.ReLU()  # Activation function for 2nd hidden layer
        self.fc3 = nn.Linear(32, 1)  # 2nd hidden layer to output layer
        self.sigmoid = nn.Sigmoid()  # Activation for binary output

        self.loss_function = nn.BCELoss()  # Binary Cross Entropy Loss
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        # Forward pass through the layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x

    def train_model(self, data, iterations):
        for epoch in range(100):  # 100 epochs
            self.optimizer.zero_grad()  # Clear gradients
            predictions = model(data)  # Forward pass
            loss = self.loss_function(predictions, torch.tensor([1, 0, 1, 0], dtype=torch.float32).unsqueeze(1))  # Target labels
            loss.backward()  # Backward pass
            self.optimizer.step()  # Update weights
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')


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
        print("X_train shape:", X_train.shape)
        print("X_val shape:", X_val.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_val shape:", y_val.shape)
        print("y_test shape:", y_test.shape)
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(model: NeuralNetwork3, X_train: torch.Tensor, y_train, X_val, y_val):
    # loss function and optimizer
    # loss_fn = nn.BCELoss()  # binary cross entropy
    # optimizer = optim.Adam(model.parameters(), lr=0.0001)

    n_epochs = 250  # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = 0  # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                start = int(start)
                # take a batch
                X_batch = (X_train[start:(start + batch_size)])
                y_batch = (y_train[start:(start + batch_size)])

                print(X_batch.shape)
                # forward pass
                y_pred = model(X_batch)
                loss = model.loss_function(y_pred, y_batch)
                # backward pass
                model.optimizer.zero_grad()
                loss.backward()
                # update weights
                model.optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using {device} device")

    # model = NeuralNetwork().to(device)
    # if debug > 0:
    #     print(model)
    #
    # X = torch.rand(10, 21, device=device)
    # logits = model(X)
    # print(f"logits: {logits}")
    # pred_probab = nn.Softmax(dim=1)(logits)
    # y_pred = pred_probab.argmax(1)
    # print(f"Predicted class: {y_pred}")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data('diabetes_binary_health_indicators_BRFSS2015.csv')

    X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    X_val = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    y_val = torch.tensor(y_val.to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)

    model = NeuralNetwork3().to(device)
    if debug > 0:
        print(model)

    train_model(model, X_train, y_train, X_val, y_val)
