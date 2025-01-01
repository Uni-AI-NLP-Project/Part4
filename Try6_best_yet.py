import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
import os
import shutil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_STATE_FILE_NAME: str = ""
TEST_ONLY = False  # Should be true if only testing
DATA_SPLIT_RANDOM_STATE = 54321

IN_LAYER_SIZE = 21
HIDDEN_LAYER_SIZES: tuple = (32, 16, 8, 4, 2)  # (128, 128, 128, 128, 128, 128, 128, 64, 32)
OUT_LAYER_SIZE = 1
POS_WEIGHT_AMOUNT = 3  # 86 / 14
DROPOUT_AMOUNT = 0.3
INIT_LEARNING_RATE = 0.001
INIT_WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 500

MAX_EPOCHS = 300000
EPOCHS_PER_VALIDATION = 100
EPOCHS_PER_SAVE_STATE = 2000
PATIENCE_LIMIT = 10000  # Early-stopping Patience

EPSILON = 1e-7
# NORMALIZE_DATA = False  # Whether to normalize each entry in the data to be in [0,1]
PREPROCESS_DIABETES_DATA = True
TRY_TO_OVERFIT = False  # Should probably be false
DEBUG = 1


# noinspection PyShadowingNames
def load_data(path_to_csv):
    data = pd.read_csv(path_to_csv)

    # if NORMALIZE_DATA:
    #     scaler = MinMaxScaler()
    #     data = scaler.fit_transform(data)
    #     X = np.delete(data, 0, axis=1)
    #     y = data[:, 0]
    if PREPROCESS_DIABETES_DATA:
        data = preprocess_diabetes_data(data).to_numpy()
    else:
        data = data.to_numpy()
    X = np.delete(data, 0, axis=1)
    y = data[:, 0]

    # Split the dataset - once for train/test, 2nd time for true_test/validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=DATA_SPLIT_RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=DATA_SPLIT_RANDOM_STATE, stratify=y_train
    )

    if DEBUG > 0:
        print(f"X_train shape: {X_train.shape}, type: {type(X_train)}")
        print("X_val shape:", X_val.shape)
        print("X_test shape:", X_test.shape)
        print("y_train shape:", y_train.shape)
        print("y_val shape:", y_val.shape)
        print("y_test shape:", y_test.shape)

    X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    X_val = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    y_val = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(DEVICE)

    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_diabetes_data(data):
    """
    Preprocesses the data - one-hot encodes categorical features.
    :param data: Data to preprocess
    :return: The preprocessed data.
    """
    # One-hot encode specified features
    one_hot_features = ['GenHlth', 'Age', 'Education', 'Income']
    encoder = OneHotEncoder(sparse_output=False, drop=None)  # drop=None keeps all categories
    encoded_data = encoder.fit_transform(data[one_hot_features])

    # Create a DataFrame for the one-hot encoded columns
    encoded_columns = encoder.get_feature_names_out(one_hot_features)
    encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

    # Drop original columns and add the one-hot encoded ones
    data = pd.concat([data.drop(columns=one_hot_features), encoded_df], axis=1)
    global IN_LAYER_SIZE
    IN_LAYER_SIZE = data.shape[1] - 1
    return data


# noinspection PyUnresolvedReferences
def con_mat(y_true, y_pred):
    """
    Returns confusion matrix:
        true negative   false positive
        false negative  true positive
    :param y_true: Real values
    :param y_pred: Predicted values
    :return:
    """
    cm = torch.zeros((2, 2), device=DEVICE)
    cm[0, 0] = ((y_true == 0) & (y_pred == 0)).sum()
    cm[0, 1] = ((y_true == 0) & (y_pred == 1)).sum()
    cm[1, 0] = ((y_true == 1) & (y_pred == 0)).sum()
    cm[1, 1] = ((y_true == 1) & (y_pred == 1)).sum()
    return cm


def f1score(_confusion_matrix):
    """
    F1 Score = (2*TP)/(2*TP + FP + FN)
    :param _confusion_matrix: A confusion matrix return from con_mat()
    :return: The f1 score of the model
    """
    tp = _confusion_matrix[1, 1]
    fp = _confusion_matrix[0, 1]
    fn = _confusion_matrix[1, 0]

    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    f1 = 2 * precision * recall / (precision + recall + EPSILON)
    return f1


class DiabetesBinaryNN(nn.Module):
    def __init__(self, input_size, hidden_layers_sizes):
        super().__init__()
        layers: list = [nn.Linear(in_features=input_size, out_features=hidden_layers_sizes[0])]
        if not TRY_TO_OVERFIT:
            layers.append(nn.BatchNorm1d(hidden_layers_sizes[0]))
        layers.append(nn.ReLU())
        if not TRY_TO_OVERFIT:
            layers.append(nn.Dropout(DROPOUT_AMOUNT))

        for i in range(len(hidden_layers_sizes) - 1):
            layers.append(nn.Linear(in_features=hidden_layers_sizes[i], out_features=hidden_layers_sizes[i + 1]))
            if not TRY_TO_OVERFIT:
                layers.append(nn.BatchNorm1d(hidden_layers_sizes[i + 1]))
            layers.append(nn.ReLU())
            if not TRY_TO_OVERFIT:
                layers.append(nn.Dropout(DROPOUT_AMOUNT))
        layers.append(nn.Linear(in_features=hidden_layers_sizes[-1], out_features=OUT_LAYER_SIZE))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# noinspection PyShadowingNames
def train_model(model: DiabetesBinaryNN, X_train, X_val, y_train, y_val, epochs, device):
    # Set up loss function (with binary class weights)
    pos_weight = torch.tensor([POS_WEIGHT_AMOUNT]).to(device)  # Keeping your original class weights
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Set up optimizer
    if TRY_TO_OVERFIT:
        optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=INIT_WEIGHT_DECAY)
    # optimizer = torch.optim.SGD(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=INIT_WEIGHT_DECAY)

    # Set up learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,
                                                           mode='min',
                                                           factor=0.5,
                                                           patience=SCHEDULER_PATIENCE)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer,
    #                                                 max_lr=INIT_LEARNING_RATE * 10,
    #                                                 steps_per_epoch=10,  # len(X_train) // batch_size,
    #                                                 epochs=epochs)

    best_val_loss = float('inf')
    best_f1v = 0
    patience_counter = 0

    for epoch in range(epochs + 1):
        model.train()

        # Forward pass
        train_logits = model(X_train).squeeze()
        train_loss = loss_fn(train_logits, y_train)

        # Validation phase
        if epoch % EPOCHS_PER_VALIDATION == 0:
            model.eval()
            with torch.inference_mode():
                val_logits = model(X_val).squeeze()
                val_loss = loss_fn(val_logits, y_val)
                val_prediction = torch.round(torch.sigmoid(val_logits))
                val_acc = (val_prediction == y_val).float().mean()
                to_print: str = f"Epoch {epoch}: Train Loss = {train_loss:.5f}, Val Loss = {val_loss:.5f}, Val Acc = {val_acc:.5f}"

                train_prediction = torch.round(torch.sigmoid(train_logits))
                train_confuision_mat = con_mat(y_train, train_prediction)
                f1 = f1score(train_confuision_mat)
                val_confuision_mat = con_mat(y_val, val_prediction)
                f1v = f1score(val_confuision_mat)
                to_print += f" | Train F1 = {f1:.5f}, Val F1 = {f1v:.5f}"
                if f1v > best_f1v:
                    best_f1v = f1v
                    patience_counter = 0
                    torch.save(model.state_dict(), f"models/best model (val F1 = {f1v:.5f}) {HIDDEN_LAYER_SIZES}.pth")
                elif val_loss < best_val_loss:  # Best validation loss check
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f"models/best model (val loss = {val_loss:.5f}) {HIDDEN_LAYER_SIZES}.pth")
                print(to_print)

        if patience_counter >= PATIENCE_LIMIT:  # Early stopping check
            print(f"Early stopping triggered ({PATIENCE_LIMIT} bad epochs)")
            return

        # Backward pass
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        # scheduler.step(train_loss)
        scheduler.step(val_loss)

        patience_counter += 1

        if epoch % EPOCHS_PER_SAVE_STATE == 0:  # Save checkpoints periodically
            torch.save(model.state_dict(), f"models/checkpoint_epoch_{epoch}.pth")


if __name__ == '__main__':
    print(f"Using {DEVICE} device")

    # Prepare directory
    if os.path.exists("models"):
        shutil.rmtree("models")
    os.mkdir("models")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data('diabetes_binary_health_indicators_BRFSS2015.csv')

    # Initialize model
    model = DiabetesBinaryNN(IN_LAYER_SIZE, HIDDEN_LAYER_SIZES).to(DEVICE)
    if MODEL_STATE_FILE_NAME != "":
        model.load_state_dict(torch.load(MODEL_STATE_FILE_NAME, weights_only=True))

    # Train model
    if not TEST_ONLY:
        train_model(model, X_train, X_val, y_train, y_val, MAX_EPOCHS, device=DEVICE)

    # Test Model
    model.eval()
    with torch.inference_mode():
        test_logits = model(X_test).squeeze()
        test_prediction = torch.round(torch.sigmoid(test_logits))
        test_acc = (test_prediction == y_val).float().mean()
        f1t = f1score(con_mat(y_test, test_prediction))
        print(f"\nTest accuracy = {test_acc:.5f}, Test F1 = {f1t:.5f}")
