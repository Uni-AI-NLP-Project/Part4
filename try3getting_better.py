import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"
in_n = 21
l1_n = 64
l2_n = 128
l3_n = 64
out_n = 1

load_file_name: str = ""
epochs = 100000

debug = 0


def load_data(path_to_csv):
    data = pd.read_csv(path_to_csv)

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Separate features (X) and target (y)
    # X = data.drop('Diabetes_binary', axis=1)
    X = np.delete(data, 0, axis=1)
    y = data[:, 0]

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

    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    return X_train, X_val, X_test, y_train, y_val, y_test


class try3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=in_n, out_features=l1_n)
        self.layer_2 = nn.Linear(in_features=l1_n, out_features=l2_n)
        self.layer_3 = nn.Linear(in_features=l2_n, out_features=l3_n)
        self.layer_4 = nn.Linear(in_features=l3_n, out_features=out_n)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

        self.bn1 = nn.BatchNorm1d(l1_n)
        self.bn2 = nn.BatchNorm1d(l2_n)
        self.bn3 = nn.BatchNorm1d(l3_n)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_4(x)
        # x = self.sigmoid(x)
        return x


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred))
    return acc


if __name__ == '__main__':
    if os.path.exists("model"):
        shutil.rmtree("model")
    os.mkdir("model")
    print(f"Using {device} device")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data('diabetes_binary_health_indicators_BRFSS2015.csv')

    model = try3().to(device)
    if load_file_name != "":
        model.load_state_dict(torch.load(load_file_name, weights_only=True))
    loss_fn = nn.BCEWithLogitsLoss()
    parameters = model.parameters()
    # optimizer = torch.optim.SGD(parameters, lr=0.001)
    optimizer = torch.optim.Adam(parameters, lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    min_loss = 100
    max_validation_acc = 0
    save_model = True
    is_best_model = False
    file_name: str = "default"
    validation_acc = None
    for epoch in range(epochs + 1):
        # Forward pass
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        # Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        # Model evaluation
        model.eval()
        with torch.inference_mode():
            validation_logits = model(X_val).squeeze()
            validation_pred = torch.round(torch.sigmoid(validation_logits))
            validation_loss = loss_fn(validation_logits, y_val)
            validation_acc = accuracy_fn(y_true=y_val, y_pred=validation_pred)
            if max_validation_acc < validation_acc:
                is_best_model = True

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.5f} "
                  f"| Validation Loss: {validation_loss:.5f}, Validation Accuracy: {validation_acc:.5f}")

        file_name = f"model/epoch{epoch}_loss{loss:.4f}_valAcc{validation_acc:.4f}"
        if min_loss > loss + 0.005:
            save_model = True
            min_loss = loss
        if is_best_model:
            save_model = True
            max_validation_acc = validation_acc
            file_name += "_best"
            is_best_model = False
        if save_model or epoch % 5000 == 0:
            torch.save(model.state_dict(), file_name)
            save_model = False

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
