import joblib
from sklearn.preprocessing import StandardScaler
import torch

def normalize_df(df, cols):
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model_class, path, *args):
    model = model_class(*args)
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
