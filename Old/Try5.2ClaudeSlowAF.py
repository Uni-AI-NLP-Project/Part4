import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import os
import shutil
from torch.optim.lr_scheduler import OneCycleLR

state_file_name: str = ""

MaxEpochs = 100000
BatchSize = 256  # Added batch training
EpochsPerValidation = 100
EpochsPerSaveState = 1000
EpochsPerF1Calc = 1000
patience_limit = 5000

def_hidden_layers_sizes = [64, 32, 16]  # Deeper network
pos_weight_amount = 80 / 20  # Adjusted class weight

epsilon = 1e-7
debug = 0


# noinspection PyShadowingNames
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


class DiabetesNetwork(nn.Module):
    def __init__(self, input_size=21, hidden_layers_sizes=None):
        super().__init__()
        if hidden_layers_sizes is None:
            hidden_layers_sizes = def_hidden_layers_sizes

        layers = []

        # Input layer with PReLU activation
        layers.extend([
            nn.Linear(input_size, hidden_layers_sizes[0]),
            nn.BatchNorm1d(hidden_layers_sizes[0]),
            nn.PReLU(),
            nn.Dropout(0.4)  # Higher dropout for first layer
        ])

        # Hidden layers with decreasing dropout
        dropout_rates = [0.3, 0.2, 0.1]  # Decreasing dropout rates
        for i in range(len(hidden_layers_sizes) - 1):
            layers.extend([
                nn.Linear(hidden_layers_sizes[i], hidden_layers_sizes[i + 1]),
                nn.BatchNorm1d(hidden_layers_sizes[i + 1]),
                nn.PReLU(),
                nn.Dropout(dropout_rates[i])
            ])

        # Output layer
        layers.append(nn.Linear(hidden_layers_sizes[-1], 1))

        self.network = nn.Sequential(*layers)

        # Initialize weights using He initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.network(x)


def create_data_loaders(X_train, X_val, y_train, y_val, batch_size):
    # Convert to PyTorch datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader


def train_model(model, X_train, X_val, y_train, y_val, epochs, device="cuda"):
    # Create data loaders
    train_loader, val_loader = create_data_loaders(X_train, X_val, y_train, y_val, BatchSize)

    # Loss function with class weights
    pos_weight = torch.tensor([pos_weight_amount]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer with increased learning rate
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)

    # OneCycle learning rate scheduler
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.002,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # Warmup for 30% of training
        div_factor=25.0,
        final_div_factor=1000.0
    )

    best_val_loss = float('inf')
    best_f1 = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        print(epoch)
        model.train()
        total_loss = 0

        # Training loop
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            y_logits = model(batch_X).squeeze()
            loss = loss_fn(y_logits, batch_y)
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation phase
        if epoch % EpochsPerValidation == 0:
            model.eval()
            val_loss = 0
            all_val_preds = []
            all_val_targets = []

            with torch.inference_mode():
                for batch_X, batch_y in val_loader:
                    val_logits = model(batch_X).squeeze()
                    val_loss += loss_fn(val_logits, batch_y).item()
                    val_preds = torch.round(torch.sigmoid(val_logits))
                    all_val_preds.extend(val_preds.cpu().numpy())
                    all_val_targets.extend(batch_y.cpu().numpy())

                val_loss /= len(val_loader)
                val_preds = np.array(all_val_preds)
                val_targets = np.array(all_val_targets)
                val_acc = (val_preds == val_targets).mean()

                to_print = f"Epoch {epoch}: Train Loss = {avg_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}"

                if epoch % EpochsPerF1Calc == 0:
                    val_confusion_mat = confusion_matrix(val_targets, val_preds)
                    f1v = f1score(val_confusion_mat)
                    to_print += f", Val F1 = {f1v:.4f}"

                    if f1v > best_f1:
                        best_f1 = f1v
                        torch.save(model.state_dict(), "model/best_f1_model.pth")

                print(to_print)

                # Early stopping check
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), "model/best_loss_model.pth")
                else:
                    patience_counter += 1
                    if patience_counter >= patience_limit:
                        print("Early stopping triggered!")
                        return

        # Save checkpoints
        if epoch % EpochsPerSaveState == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'best_f1': best_f1
            }, f"model/checkpoint_epoch_{epoch}.pth")


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    if os.path.exists("model"):
        shutil.rmtree("model")
    os.mkdir("model")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data('../diabetes_binary_health_indicators_BRFSS2015.csv')

    model = DiabetesNetwork().to(device)
    if state_file_name != "":
        model.load_state_dict(torch.load(state_file_name, weights_only=True))

    train_model(model, X_train, X_val, y_train, y_val, MaxEpochs, device=device)
