import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
import os
import shutil

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

STATE_FILE_NAME: str = ""
DATA_SPLIT_RANDOM_STATE = 1

HIDDEN_LAYER_SIZES: tuple = (1,)  # (128, 64, 32)
POS_WEIGHT_AMOUNT = 86 / 14
DROPOUT_AMOUNT = 0.3
INIT_LEARNING_RATE = 0.001
INIT_WEIGHT_DECAY = 1e-5
SCHEDULER_PATIENCE = 1000

MAX_EPOCHS = 100000
EPOCHS_PER_VALIDATION = 100
EPOCHS_PER_SAVE_STATE = 2000
# EPOCHS_PER_F1_CALC = 100
PATIENCE_LIMIT = 100000  # Early-stopping Patience

DEBUG = 0
EPSILON = 1e-7
NORMALIZE_DATA = True  # Whether to normalize each entry in the data to be in [0,1]
TRY_TO_OVERFIT = False  # Should probably be false


# noinspection PyShadowingNames
def load_data(path_to_csv):
    data = pd.read_csv(path_to_csv)

    if NORMALIZE_DATA:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        X = np.delete(data, 0, axis=1)
        y = data[:, 0]
    else:
        data_np = data.to_numpy()
        X = np.delete(data_np, 0, axis=1)
        y = data_np[:, 0]

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


def analyze_feature_importance(X, y):
    """
    Analyze feature importance using multiple methods
    Returns: Dictionary of importance scores for each feature
    """
    # 1. Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_

    # 2. Mutual Information
    mi_importance = mutual_info_classif(X, y)

    # 3. Correlation with target (absolute value)
    X_df = pd.DataFrame(X)
    correlations = np.abs(X_df.corrwith(pd.Series(y)))

    # Combine scores (normalize each method's scores first)
    _importance_scores = (rf_importance / rf_importance.max() +
                          mi_importance / mi_importance.max() +
                          correlations / correlations.max()) / 3

    return _importance_scores


def initialize_feature_weights(_model, importance_scores):
    """
    Initialize the network's feature weights based on importance scores
    """
    with torch.no_grad():
        _model.feature_weights.data = torch.tensor(importance_scores,
                                                   dtype=torch.float32
                                                   ).to(_model.feature_weights.device)


class DiabetesBinaryNN(nn.Module):
    def __init__(self, input_size=21, hidden_layers_sizes=HIDDEN_LAYER_SIZES):
        super().__init__()
        self.feature_weights = nn.Parameter(torch.ones(input_size))
        layers: list = [nn.Linear(in_features=input_size, out_features=hidden_layers_sizes[0])]
        if not TRY_TO_OVERFIT:
            layers.append(nn.BatchNorm1d(hidden_layers_sizes[0]))
        # layers.append(nn.ReLU())
        layers.append(nn.PReLU())
        if not TRY_TO_OVERFIT:
            layers.append(nn.Dropout(DROPOUT_AMOUNT))

        for i in range(len(hidden_layers_sizes) - 1):
            layers.append(nn.Linear(in_features=hidden_layers_sizes[i], out_features=hidden_layers_sizes[i + 1]))
            if not TRY_TO_OVERFIT:
                layers.append(nn.BatchNorm1d(hidden_layers_sizes[i + 1]))
            # layers.append(nn.ReLU())
            layers.append(nn.PReLU())
            if not TRY_TO_OVERFIT:
                layers.append(nn.Dropout(DROPOUT_AMOUNT))
        layers.append(nn.Linear(in_features=hidden_layers_sizes[-1], out_features=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # return self.network(x)
        weighted_input = x * self.feature_weights
        return self.network(weighted_input)


# noinspection PyShadowingNames
def train_model(model, X_train, X_val, y_train, y_val, epochs, device):
    # Set up loss function (with binary class weights)
    pos_weight = torch.tensor([POS_WEIGHT_AMOUNT]).to(device)  # Keeping your original class weights
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Set up optimizer
    if TRY_TO_OVERFIT:
        optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LEARNING_RATE, weight_decay=INIT_WEIGHT_DECAY)

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
    best_f1 = False
    best_f1v = 0
    patience_counter = 0

    for epoch in range(epochs + 1):
        model.train()

        # Forward pass
        y_logits = model(X_train).squeeze()
        loss = loss_fn(y_logits, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step(loss)
        scheduler.step(loss)

        # Validation phase
        if epoch % EPOCHS_PER_VALIDATION == 0:
            model.eval()
            with torch.inference_mode():
                val_logits = model(X_val).squeeze()
                val_loss = loss_fn(val_logits, y_val)
                val_preds = torch.round(torch.sigmoid(val_logits))
                val_acc = (val_preds == y_val).float().mean()
                to_print: str = f"Epoch {epoch}: Train Loss = {loss:.5f}, Val Loss = {val_loss:.5f}, Val Acc = {val_acc:.5f}"

                # if epoch % EPOCHS_PER_F1_CALC == 0:
                y_preds = torch.round(torch.sigmoid(y_logits))
                train_confuision_mat = con_mat(y_train, y_preds)
                f1 = f1score(train_confuision_mat)
                val_confuision_mat = con_mat(y_val, val_preds)
                f1v = f1score(val_confuision_mat)
                to_print += f" | Train F1 = {f1:.5f}, Val F1 = {f1v:.5f}"
                if f1v > best_f1v:
                    best_f1v = f1v
                    best_f1 = True
                print(to_print)

                # Learning rate scheduling
                # scheduler.step(val_loss)

                if best_f1:  # Best F1 score check
                    patience_counter = 0
                    torch.save(model.state_dict(), f"model/best model (val F1 = {f1v:.5f}).pth")
                    # save_model(model.state_dict(),
                    #            optimizer.state_dict(),
                    #            f"model/best model (val F1 = {f1v:.5f}).pth")
                    best_f1 = False
                elif val_loss < best_val_loss:  # Best loss check
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), f"model/best model (val loss = {val_loss:.5f}).pth")
                    # save_model(model.state_dict(),
                    #            optimizer.state_dict(),
                    #            f"model/best model (val loss = {val_loss:.5f}).pth")
                elif patience_counter >= PATIENCE_LIMIT:  # Early stopping check
                    print(f"Early stopping triggered ({PATIENCE_LIMIT} bad epochs)")
                    return
        patience_counter += 1

        # Save checkpoints periodically
        if epoch % EPOCHS_PER_SAVE_STATE == 0:
            torch.save(model.state_dict(), f"model/checkpoint_epoch_{epoch}.pth")
            # torch.save({
            #     'epoch': epoch,
            #     'model_state_dict': model.state_dict(),
            #     'optimizer_state_dict': optimizer.state_dict(),
            #     'loss': loss,
            # }, f"model/checkpoint_epoch_{epoch}.pth")
            # save_model(model.state_dict(),
            #            optimizer.state_dict(),
            #            f"model/checkpoint_epoch_{epoch}.pth")


def save_model(model_state_dict, optimizer_state_dict, file_name: str):
    torch.save({
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
    }, file_name)


if __name__ == '__main__':
    print(f"Using {DEVICE} device")

    # Prepare directory
    if os.path.exists("model"):
        shutil.rmtree("model")
    os.mkdir("model")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data('../diabetes_binary_health_indicators_BRFSS2015.csv')

    print("Weighing features")
    X_numpy = X_train.cpu().numpy()
    y_numpy = y_train.cpu().numpy()
    importance_scores = analyze_feature_importance(X_numpy, y_numpy)

    # Initialize model
    # model = DiabetesBinaryNN().to(DEVICE)
    model = DiabetesBinaryNN().to(DEVICE)
    initialize_feature_weights(model, importance_scores)
    if STATE_FILE_NAME != "":
        model.load_state_dict(torch.load(STATE_FILE_NAME, weights_only=True))

    # Train model
    print("Starting training")
    train_model(model, X_train, X_val, y_train, y_val, MAX_EPOCHS, device=DEVICE)
