import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"
in_n = 21
l1_n = 42
l2_n = 84
# l3_n = 84
# l4_n = 84
# l5_n = 42
# l6_n = 21
out_n = 1

load_file_name: str = ""
epochs = 100000
con_matrix = True  # SLOWS TRAINING IF SET TRUE SINCE FUNCTION WILL MOVE STUFF BACK TO CPU

epsilon = 0.000001
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


class try4(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_layer = nn.Linear(in_features=in_n, out_features=l1_n)
        self.hlayer_1 = nn.Linear(in_features=l1_n, out_features=l2_n)
        self.hlayer_2 = nn.Linear(in_features=l2_n, out_features=out_n)
        # self.hlayer_2 = nn.Linear(in_features=l2_n, out_features=l3_n)
        # self.hlayer_3 = nn.Linear(in_features=l3_n, out_features=l4_n)
        # self.hlayer_4 = nn.Linear(in_features=l4_n, out_features=l5_n)
        # self.hlayer_5 = nn.Linear(in_features=l5_n, out_features=l6_n)
        # self.out_layer = nn.Linear(in_features=l6_n, out_features=out_n)
        # self.out_layer = nn.Linear(in_features=l2_n, out_features=out_n)

        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        # self.dropout = nn.Dropout(p=0.5)

        # self.bn1 = nn.BatchNorm1d(l1_n)
        # self.bn2 = nn.BatchNorm1d(l2_n)
        # self.bn3 = nn.BatchNorm1d(l3_n)

    def forward(self, x):
        x = self.in_layer(x)
        x = self.relu(x)

        x = self.hlayer_1(x)
        x = self.relu(x)

        x = self.hlayer_2(x)
        # x = nn.ReLU(x)
        #
        # x = self.hlayer_3(x)
        # x = nn.ReLU(x)
        #
        # x = self.hlayer_4(x)
        # x = nn.ReLU(x)
        #
        # x = self.hlayer_5(x)
        # x = nn.ReLU(x)
        #
        # x = self.out_layer(x)
        # # x = self.sigmoid(x)
        return x


def con_mat(y_true, y_pred):
    """
    Returns confusion matrix:
        true negative   false positive
        false negative  true positive
    :param y_true:
    :param y_pred:
    :return:
    """
    true = y_true.cpu().detach().numpy()
    pred = y_pred.cpu().detach().numpy()
    ans = confusion_matrix(true, pred)
    return ans


def f1score(_confusion_matrix):
    """
    F1 Score = (2*TP)/(2*TP + FP + FN)
    :param _confusion_matrix:
    :return:
    """
    tp = _confusion_matrix[1, 1]
    fp = _confusion_matrix[0, 1]
    fn = _confusion_matrix[1, 0]

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    return 2 * precision * recall / (precision + recall + epsilon)


def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred))
    return acc


if __name__ == '__main__':
    if os.path.exists("model"):
        shutil.rmtree("model")
    os.mkdir("model")
    print(f"Using {device} device")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data('../diabetes_binary_health_indicators_BRFSS2015.csv')

    model = try4().to(device)
    if load_file_name != "":
        model.load_state_dict(torch.load(load_file_name, weights_only=True))

    pos_weight = torch.tensor([86 / 14]).to(device)  # Higher weight for positives (diabetes = 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    parameters = model.parameters()
    # optimizer = torch.optim.SGD(parameters, lr=0.001)
    optimizer = torch.optim.Adam(parameters, lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)

    min_loss = 100
    max_validation_acc = 0
    max_f1 = 0
    max_f1v = 0
    save_model = True
    is_best_val = False
    is_best_f1v = False
    file_name: str = "default"
    validation_acc = None
    for epoch in range(epochs + 1):
        # Forward pass
        y_logits = model(X_train).squeeze()
        y_pred = torch.round(torch.sigmoid(y_logits))

        # Calculate loss and accuracy
        loss = loss_fn(y_logits, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred)

        # Model evaluation and save
        file_name = f"model/epoch{epoch}_loss{loss:.5f}"
        if epoch % 100 == 0:  # Validates only once in "a while
            model.eval()
            with torch.inference_mode():
                validation_logits = model(X_val).squeeze()
                validation_pred = torch.round(torch.sigmoid(validation_logits))
                validation_loss = loss_fn(validation_logits, y_val)
                validation_acc = accuracy_fn(y_true=y_val, y_pred=validation_pred)
                if con_matrix:
                    confuision_mat = con_mat(y_train, y_pred)
                    val_confuision_mat = con_mat(y_val, validation_pred)
                    f1 = f1score(confuision_mat)
                    f1v = f1score(val_confuision_mat)
                    # print(confuision_mat)
                    f1avg = (f1 + f1v) / 2
                    if max_f1v < f1v:
                        is_best_f1v = True
                        max_f1v = f1v
                        file_name += "_best_val_f1"
                    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.5f} "
                          f"| V-Loss: {validation_loss:.5f}, V-Acc: {validation_acc:.5f} | "
                          f"F1: {f1:.5f}, V-F1:{f1v:.5f}, Avg: {f1avg:.5f}")
                else:
                    print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.5f} "
                          f"| V-Loss: {validation_loss:.5f}, V-Acc: {validation_acc:.5f}")
                if max_validation_acc < validation_acc:
                    is_best_val = True

        if is_best_f1v:
            save_model = True
            is_best_f1v = False
        if con_matrix and f1 > max_f1:
            save_model = True
            max_f1 = f1
            file_name += "_best_f1"
        if is_best_val:
            save_model = True
            max_validation_acc = validation_acc
            file_name += "_best_val_acc"
            is_best_val = False
        if min_loss > loss + 0.005:
            save_model = True
            min_loss = loss
        if save_model or epoch % 5000 == 0:
            torch.save(model.state_dict(), file_name)
            save_model = False

        # Backward pass
        # optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
