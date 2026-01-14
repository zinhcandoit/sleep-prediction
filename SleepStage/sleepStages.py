#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import library
import pandas as pd
import numpy as np
import os
import glob
import csv
import random


# ## Extract and group file by id

# In[2]:


folder_path = ["heart_rate/", "motion/", "labels/"]
files = glob.glob(folder_path[2]+"*.csv")
files_name = [os.path.basename(p) for p in files]
id_list = []
for f in files_name:
    id = f.split("_")[0]
    id_list.append(id)
print(id_list)
print(len(id_list))


# In[ ]:


files_by_id = {id_code: [] for id_code in id_list}

for folder in folder_path:
    txt_files = glob.glob(os.path.join(folder, "*.csv"))

    for file_path in txt_files:
        file_name = os.path.basename(file_path)

        # Tách ID (trước "_" hoặc ".")
        id_in_file = file_name.split("_")[0].split(".")[0]

        if id_in_file in id_list:
            files_by_id[id_in_file].append(file_path)


# ### Process for Heart Rate

# In[4]:


HR_list = []

for id_code, paths in files_by_id.items():
    df_HR = pd.read_csv(paths[0])

    df_HR["timestamp"] = pd.to_numeric(df_HR["timestamp"], errors="coerce")
    df_HR = df_HR[df_HR["timestamp"] >= 0].dropna(subset=["timestamp"])
    HR_list.append(df_HR)


# In[5]:


for data in HR_list:
    print(data.head())


# In[6]:


# Normalize label
label_list = []
for id_code, paths in files_by_id.items():
    df_l = pd.read_csv(paths[2])
    df_l["Sleepstage"] = (
        pd.to_numeric(df_l["Sleepstage"], errors="coerce")
        .replace(-1, np.nan)
        .replace(5, 4)
        .fillna(
            df_l["Sleepstage"]
            .rolling(window=3, center=True)
            .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
        )
        .ffill()
        .bfill()
        .astype(int)
    )
    label_list.append(df_l)


# In[7]:


# Normalize motion
motion_list = []
for id_code, paths in files_by_id.items():
    df_m = pd.read_csv(paths[1])

    df_m["timestamp"] = pd.to_numeric(df_m["timestamp"], errors="coerce")
    df_m = df_m[df_m["timestamp"] >= 0].dropna(subset=["timestamp"])
    motion_list.append(df_m)


# In[8]:


# Merge
merge_list = []
for i in range(len(id_list)):
    df_h = HR_list[i] 
    df_l = label_list[i]
    df_m = motion_list[i]

    df_h["timestamp"] = pd.to_datetime(df_h["timestamp"], unit="s")
    df_m["timestamp"] = pd.to_datetime(df_m["timestamp"], unit="s")
    df_l["timestamp"] = pd.to_datetime(df_l["timestamp"], unit="s")

    re_df_h = df_h.set_index("timestamp").resample("30s").mean().reset_index()
    re_df_m = df_m.set_index("timestamp").resample("30s").mean().reset_index()
    re_df_l = df_l.copy()

    df_merge = re_df_l.merge(re_df_h, on=["timestamp"], how="left")
    df_merge = df_merge.merge(re_df_m, on=["timestamp"], how="left")
    df_merge["timestamp"] = df_merge["timestamp"].dt.hour*3600 + df_merge["timestamp"].dt.minute*60 + df_merge["timestamp"].dt.second
    
    merge_list.append(df_merge)

print(merge_list[5].head())


# In[9]:


for df in merge_list:
    s = pd.to_numeric(df["Sleepstage"], errors="coerce")
    s = s.replace({-1: np.nan, 5: 4})

    s_filled = s.fillna(
        s.rolling(window=5, center=True)
         .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan, raw=False)
    )

    s_filled = s_filled.ffill().bfill()

    df["Sleepstage"] = s_filled.astype(int)

    assert not (df["Sleepstage"] == -1).any(), "Sleepstage vẫn còn -1"

    #HR
    df["HR"] = (
        df["HR"]
        .fillna(df["HR"].rolling(window=5, center=True).median())
        .ffill()
        .bfill()
    )

    #Accelerometer
    for col in ["ax", "ay", "az"]:
        df[col] = (
            df[col]
            .fillna(df[col].rolling(window=5, center=True).median())
            .ffill()
            .bfill()
        )


# In[10]:


s = 0
for df in merge_list:
    s+= df[['timestamp', "Sleepstage", "HR", "ax", "ay", "az"]].isna().sum()
print(s)
merge_list[0].head()


# In[11]:


def normalize_axis(df):
    df[['dax','day','daz']] = (
        df[['ax', 'ay', 'az']]
        .diff()
        .fillna(0)
    )
    return df

def customize_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.sort_values("timestamp").reset_index(drop=True)

    normalize_axis(df)

    df['dacc_mag'] = np.sqrt(df['dax']**2 + df['day']**2 + df['daz']**2)

    # Timestamp by minutes
    df["timestamp"] = df["timestamp"] / 60.0

    # RR interval (ms)
    df["RR"] = 60000.0 / df["HR"].replace(0, np.nan)
    df["RR"] = df["RR"].ffill().bfill()

    # RR diff
    df["RR_diff"] = df["RR"].diff().fillna(0)

    win = 20    # 1 epoch = 30s, window = 600s = 10 min

    # RMSSD
    df["HRV_RMSSD"] = (
        df["RR_diff"].rolling(window=win, min_periods=1)
        .apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
    )

    df["acc_mean_10"] = df["dacc_mag"].rolling(window=win, min_periods=1).mean()
    df["acc_std_10"]  = df["dacc_mag"].rolling(window=win, min_periods=1).std().fillna(0)    # Standard deviation

    return df


# In[12]:


# áp dụng cho 31 người
merge_list = [customize_features(df) for df in merge_list]
print(merge_list[5].head())


# ## Prepare data for model

# In[13]:


data_list = [df for df in merge_list]


# ### Normalization

# In[14]:


for i in range(len(data_list)):
    data_list[i] = data_list[i].sort_values("timestamp").reset_index(drop=True)

for i, df in enumerate(data_list):
    df["subject_id"] = i


# In[15]:


feature_cols = [c for c in data_list[0].columns 
                if c not in ["Sleepstage", "subject_id", "timestamp"]]
print(len(feature_cols))
print(feature_cols)


# ## **Split train/val/test**

# In[16]:

ids = list(range(len(data_list)))
random.seed(42)
random.shuffle(ids)

# Split train/test/val

train_ratio = 0.7
val_ratio = 0.15


train_size = int(len(ids) * 0.70)
val_size   = int(len(ids) * 0.15)

train_ids = ids[:train_size]
val_ids   = ids[train_size:train_size + val_size]
test_ids  = ids[train_size + val_size:]

train_df = pd.concat([data_list[i] for i in train_ids])
val_df   = pd.concat([data_list[i] for i in val_ids])
test_df  = pd.concat([data_list[i] for i in test_ids])

print (test_df.describe())


# In[17]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(train_df[feature_cols])

train_df[feature_cols] = scaler.transform(train_df[feature_cols])
val_df[feature_cols]   = scaler.transform(val_df[feature_cols])
test_df[feature_cols]  = scaler.transform(test_df[feature_cols])

train_df[feature_cols] = train_df[feature_cols].interpolate(method='linear', limit_direction='both')
val_df[feature_cols] = val_df[feature_cols].interpolate(method='linear', limit_direction='both')
test_df[feature_cols] = test_df[feature_cols].interpolate(method='linear', limit_direction='both')


# In[18]:


X_train_df = train_df[feature_cols].copy()
y_train_df = train_df["Sleepstage"].copy()

X_val_df = val_df[feature_cols].copy()
y_val_df = val_df["Sleepstage"].copy()

X_test_df = test_df[feature_cols].copy()
y_test_df = test_df["Sleepstage"].copy()


# In[19]:


def df_to_sequences(df, feature_cols, T=20):
    X_seqs = []
    y_seqs = []

    for sid in df["subject_id"].unique():
        d = df[df["subject_id"] == sid].reset_index(drop=True)

        X = d[feature_cols].values
        y = d["Sleepstage"].values

        for i in range(len(d) - T + 1):
            X_seqs.append(X[i:i+T])
            y_seqs.append(y[i:i+T])   # many-to-many
    return np.array(X_seqs), np.array(y_seqs)

T = 20

X_train, y_train = df_to_sequences(train_df, feature_cols, T)
X_val, y_val     = df_to_sequences(val_df, feature_cols, T)
X_test, y_test   = df_to_sequences(test_df, feature_cols, T)


# ## **Training model**

# In[20]:


import torch.nn as nn
class SleepPrediction(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64,
                 num_classes=5, dropout1=0.3, dropout2=0.2):
        super(SleepPrediction, self).__init__()

        # CNN transforms features -> 64 channels
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # IMPORTANT: LSTM input must be 64 now
        self.lstm1 = nn.LSTM(64, hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout1)

        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout2)

        self.fc1 = nn.Linear(hidden2, 64)
        self.relu = nn.ReLU()

        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (B, T, F)
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)

        x, _ = self.lstm1(x)
        x = self.dropout1(x)

        x, _ = self.lstm2(x)
        x = self.dropout2(x)

        center = x.size(1) // 2
        x = x[:, center, :]          # (B, hidden2)

        x = self.relu(self.fc1(x))
        x = self.fc_out(x)           # (B, 5)

        return x


# In[21]:


import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)

X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

print("Train:", X_train_t.shape, y_train_t.shape)
print("Val:",   X_val_t.shape,   y_val_t.shape)
print("Test:", X_test_t.shape, y_test_t.shape)


# In[22]:


from collections import Counter

# lấy label trung tâm
y_center = y_train[:, y_train.shape[1] // 2]

counts = Counter(y_center)
num_classes = 5
total = sum(counts.values())

weights = torch.tensor(
    [total / (num_classes * counts[i]) for i in range(num_classes)],
    dtype=torch.float32
).to(device)

print("Class weights:", weights)


# In[23]:


input_dim  = X_train_t.shape[2]
num_classes = 5

model = SleepPrediction(input_dim=input_dim,
                     hidden1=128,
                     hidden2=64,
                     num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


# In[24]:


def train_loop(model, optimizer, criterion, 
               X_train, y_train,
               X_val=None, y_val=None,
               epochs=50, batch_size=64,
               patience=5):

    device = next(model.parameters()).device
    
    N = X_train.shape[0]
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(1, epochs+1):

        # Training
        idx = torch.randperm(N, device=device)
        model.train()
        
        total_loss = 0
        count = 0

        for i in range(0, N, batch_size):
            batch_idx = idx[i:i+batch_size]

            batch_x = X_train[batch_idx]     # (B, T, F)
            batch_y = y_train[batch_idx]     # (B, T)

            # Lấy label trung tâm
            center = batch_y.size(1) // 2
            batch_y_center = batch_y[:, center]   # (B,)

            optimizer.zero_grad()
            out = model(batch_x)             # (B, num_classes)

            loss = criterion(out, batch_y_center)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

        avg_train_loss = total_loss / count

        # Validation
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                out_val = model(X_val)      # (N_val, num_classes)
                center = y_val.size(1) // 2
                y_val_center = y_val[:, center]
                
                val_loss = criterion(out_val, y_val_center).item()

            print(f"Epoch {epoch}/{epochs} "
                  f"- train_loss: {avg_train_loss:.4f} "
                  f"- val_loss: {val_loss:.4f}")

            # ---- Early stopping ----
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = model.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        else:
            print(f"Epoch {epoch}/{epochs} - train_loss: {avg_train_loss:.4f}")

    # ---- restore best model ----
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Best model restored (val_loss: {best_val_loss:.4f})")
    
    return model


# In[25]:


print("Unique train labels:", np.unique(y_train))
print("Unique val labels:", np.unique(y_val))
print("Unique test labels:", np.unique(y_test))


# In[26]:


trained_model = train_loop(
    model=model,
    optimizer=optimizer,
    criterion=criterion,
    X_train=X_train_t,
    y_train=y_train_t,
    X_val=X_val_t,
    y_val=y_val_t,
    epochs=50,
    batch_size=64,
    patience=30
)


# In[27]:


from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
def test_model(model, X_test, y_test, device=None, return_preds=False):
    """
    Evaluate sleep-stage model using center label of each window.

    Parameters
    ----------
    model : nn.Module
        Trained PyTorch model (output: (B, num_classes))
    X_test : np.ndarray or torch.Tensor
        Shape (N, T, F)
    y_test : np.ndarray or torch.Tensor
        Shape (N, T)
    device : torch.device, optional
        If None, infer from model
    return_preds : bool
        If True, return y_true and y_pred

    Returns
    -------
    metrics : dict
        accuracy, classification_report, confusion_matrix
    """

    # ---- device ----
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # ---- tensor conversion ----
    if not torch.is_tensor(X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    if not torch.is_tensor(y_test):
        y_test = torch.tensor(y_test, dtype=torch.long)

    X_test = X_test.to(device)
    y_test = y_test.to(device)

    # ---- center label ----
    T = y_test.size(1)
    center = T // 2
    y_true = y_test[:, center]     # (N,)

    # ---- forward ----
    with torch.no_grad():
        logits = model(X_test)     # (N, num_classes)
        y_pred = torch.argmax(logits, dim=1)

    # ---- metrics ----
    y_true_np = y_true.cpu().numpy()
    y_pred_np = y_pred.cpu().numpy()

    metrics = {
        "accuracy": accuracy_score(y_true_np, y_pred_np),
        "classification_report": classification_report(
            y_true_np, y_pred_np, digits=4
        ),
        "confusion_matrix": confusion_matrix(y_true_np, y_pred_np)
    }

    if return_preds:
        return metrics, y_true_np, y_pred_np

    return metrics


# In[28]:


metrics = test_model(
    model=model,
    X_test=X_test,
    y_test=y_test,
)

print("Accuracy:", metrics["accuracy"])
print(metrics["classification_report"])


# In[ ]:


import joblib

MODEL_META = {
    "feature_cols": feature_cols,
    "T": T
}

torch.save({
    "model_state": model.state_dict(),
    "input_dim": input_dim,
    "meta": MODEL_META
}, "sleep_model.pth")

joblib.dump(scaler, "scaler.pkl")


