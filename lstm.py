from scipy.io import loadmat
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# ----------- LOAD, TRAIN, SPLIT DATA ------------

x = loadmat('Xtrain.mat')
x_final = loadmat('Xtest.mat')
x_test_vals = x_final['Xtest']

# Train and val size etc. subject to change
x_train = x["Xtrain"]
# x_test = x["Xtrain"][800:] # Added when test set is released

# Scale the data
scaler = StandardScaler()
x_train_2d = np.array(x_train).reshape(-1, 1)

# Fit and transform
x_train_scaled = scaler.fit_transform(x_train_2d).flatten()

#actual_test_vals = x_final['Xtest'].flatten() # extract Xtest vals 
# print(f'actual_vals: {actual_test_vals}')

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
    best_state = {k: v.clone() for k, v, in model.state_dict().items()}

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
#Thank you pookie
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

# for i in range(n_splits):
    # total_points_split = int((len(x_train_scaled) / n_splits) * (i + 1))
    # n_train_seq = int(total_points_split * proportion_train)

    # # Build sequences over the FULL split so val windows overlap the train boundary
    # X_all, y_all = create_sequences(x_train_scaled[:total_points_split], window_size)

    # X_train_np, y_train_np = X_all[:n_train_seq], y_all[:n_train_seq]
    # X_val_np,   y_val_np   = X_all[n_train_seq:], y_all[n_train_seq:]

#     splits.append((X_train_np, y_train_np, X_val_np, y_val_np))

# Sanity check for split sizes
for i in splits:
    print(len(i[0]), len(i[2]))
losses = []

# ------------------- DO CROSSVALIDATION TO FIND BEST HYPERPARAMETER ---------------
def find_optional_hyperparameters(window_options):
    best_avg_loss = float('inf')
    best_window = 10

    for w in window_options:
        split_losses = []
        print(f"Testing Window Size: {w}")
        
        for i in range(n_splits):
            total_points_split = int((len(x_train_scaled) / n_splits) * (i + 1))
            n_train_seq = int(total_points_split * proportion_train)

            # Build sequences over the FULL split so val windows overlap the train boundary
            X_all, y_all = create_sequences(x_train_scaled[:total_points_split], w)

            X_train_np, y_train_np = X_all[:n_train_seq], y_all[:n_train_seq]
            X_val_np,   y_val_np   = X_all[n_train_seq:], y_all[n_train_seq:]

            X_t = torch.FloatTensor(X_train_np).unsqueeze(-1)
            y_t = torch.FloatTensor(y_train_np)
            X_v = torch.FloatTensor(X_val_np).unsqueeze(-1)
            y_v = torch.FloatTensor(y_val_np)
            model = LSTMModel(hidden_size=64)

            val_loss = train_model(model, X_t, y_t, X_v, y_v, epochs=100)
            split_losses.append(val_loss)

        avg_loss = np.mean(split_losses)
        print(f"Average Val Loss for window {w}: {avg_loss:.6f}")

        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_window = w

    #Best Window is 30
    print(f"Optimial window size fouund: {best_window}")
    return best_window

best_final_window = find_optional_hyperparameters(window_options=[5,10,20,30,50])

# -------------------- TRAIN MODEL -------------------------------------------------
def get_final_trained_model(x_scaled, window_size, hidden = 64):
    
    X_all , y_all = create_sequences(x_scaled, window_size)
    n_val = int(len(X_all)*0.15)

    X_t = torch.FloatTensor(X_all[:-n_val]).unsqueeze(-1)
    y_t = torch.FloatTensor(y_all[: -n_val])
    X_v = torch.FloatTensor(X_all[-n_val: ]).unsqueeze(-1)
    y_v = torch.FloatTensor(y_all[-n_val: ])

    #Initialize final model
    final_model = LSTMModel(hidden_size=hidden)
    print("Starting Final Training on full dataset")
    train_model(final_model, X_t, y_t, X_v, y_v, epochs=150)
    print("Final Model Ready")
    return final_model

final_lstm = get_final_trained_model(x_train_scaled, window_size= best_final_window)

# ------------------------- PREDICT NEXT 200 STEPS -------------------------------
def predict_recursion(model, data_source, window_size, steps = 200):
    model.eval()
    predictions = []

    #Start with last window of known data
    current_window = torch.FloatTensor(data_source[-window_size:]).view(1, -1, 1)
    
    with torch.no_grad():
        for _ in range(steps):
            #Predict one step ahead

            pred = model(current_window)
            predictions.append(pred.item())

            #Update sliding window
            new_point = pred.view(1, 1, 1)
            current_window = torch.cat((current_window[:, 1:, :], new_point), dim=1)

    return np.array(predictions)

recursive_scaled = predict_recursion(final_lstm, x_train_scaled, window_size=best_final_window)

predictions_final = scaler.inverse_transform(recursive_scaled.reshape(-1,1))

# -------------------------- FIND MAE AND MSE ------------------------------------

# print(f"predictions_final", predictions_final) #check prediction final

def calculate_error(predictions, actual):
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)

    print(f'Mean Absolute Error: {mae: .5f}')
    
    print(f'Mean Squared Error: {mse: .5f}')

    plt.figure(figsize = (12,5))
    plt.plot(actual, label = 'Real measurement', color = 'skyblue')
    plt.plot(predictions, label = 'Recursive prediction', color = 'red', linestyle = '--')
    plt.xlabel('time stamps')
    plt.ylabel('value')
    plt.title('LSTM Laser predictions vs Real values')
    plt.legend()
    plt.show()

#x_test_original = scaler.inverse_transform(x_test_scaled.reshape(-1,1))
actual_test_vals = x_final['Xtest'].flatten().astype(float)

calculate_error(predictions_final, actual_test_vals)