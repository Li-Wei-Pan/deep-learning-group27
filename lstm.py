from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ----------- LOAD, TRAIN, SPLIT DATA ------------

x = loadmat('Xtrain.mat')

# Train and val size etc. subject to change
x_train = x["Xtrain"][:800]
x_test = x["Xtrain"][800:] # Added when test set is released

# Scale the data
scaler = StandardScaler()
x_train_2d = np.array(x_train).reshape(-1, 1)
x_test_2d = np.array(x_test).reshape(-1, 1)
# Fit and transform
x_train_scaled = scaler.fit_transform(x_train_2d).flatten()
x_test_scaled = scaler.transform(x_test_2d).flatten()

# --------------- DEFINE LSTM --------------

class LSTMModel(nn.Module):
    def __init__(self, hidden_size=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last timestep
        return self.fc(out)

def train_model(model, X_train, y_train, X_val, y_val, epochs=200, patience=15):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(X_train).squeeze()
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze()
            val_loss = criterion(val_pred, y_val)

        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: train={loss.item():.4f}, val={val_loss.item():.4f}")

    model.load_state_dict(best_state)
    return best_val_loss

# ---------------- DEFINE AND TUNE WINDOW SIZES PARAMETER _____________----

# Here you go amogh you got this
window_size = 10

# ----------------- SPLIT DATA  ACCORDING TO EXPANDING WINDOW SPLIT AND SEQUENCE ACCORDING TO WINDOW SIZE --------------------------
# https://medium.com/@mouadenna/time-series-splitting-techniques-ensuring-accurate-model-validation-5a3146db3088
# These splits are used for cross validation basically

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

# Number of splits
n_splits = 4
proportion_train = 0.8
splits = []

for i in range(n_splits):
    total_points_split = int((len(x_train_scaled) / n_splits) * (i + 1))
    n_train_seq = int(total_points_split * proportion_train)

    # Build sequences over the FULL split so val windows overlap the train boundary
    X_all, y_all = create_sequences(x_train_scaled[:total_points_split], window_size)

    X_train_np, y_train_np = X_all[:n_train_seq], y_all[:n_train_seq]
    X_val_np,   y_val_np   = X_all[n_train_seq:], y_all[n_train_seq:]

    splits.append((X_train_np, y_train_np, X_val_np, y_val_np))

# Sanity check for split sizes
for i in splits:
    print(len(i[0]), len(i[2]))
losses = []

# ------------------- DO CROSSVALIDATION TO FIND BEST HYPERPARAMETER ---------------

# -------------------- TRAIN MODEL -------------------------------------------------

# ------------------------- PREDICT NEXT 200 STEPS -------------------------------

# -------------------------- FIND MAE AND MSE ------------------------------------