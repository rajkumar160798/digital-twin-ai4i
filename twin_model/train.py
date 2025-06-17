import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from twin_model.model import BiGRUWithAttention
from twin_model.utils import save_model
from sklearn.preprocessing import StandardScaler
import os

# Load processed data
df = pd.read_csv("data/processed/ai4i_cleaned.csv")
sensor_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# Use only healthy samples
df_healthy = df[df['Machine failure'] == 0].reset_index(drop=True)
data = df_healthy[sensor_cols].values

# Create a PyTorch Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data, sequence_length=20):
        self.sequence_length = sequence_length
        self.data = torch.tensor(data, dtype=torch.float32)
        
    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        return (
            self.data[idx:idx+self.sequence_length],
            self.data[idx+1:idx+self.sequence_length+1]
        )

dataset = TimeSeriesDataset(data, sequence_length=20)

# Define training function
def train_gru(model, dataset, epochs=10, batch_size=32, lr=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for seq, target in dataloader:
            optimizer.zero_grad()
            pred = model(seq)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(dataloader):.4f}")
    print("Training complete.")

# Instantiate and train the model
model = BiGRUWithAttention(input_size=len(sensor_cols), hidden_size=64)
train_gru(model, dataset, epochs=10, batch_size=32)

# Save the trained model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/bigru_attention_twin.pth")
print("✅ Model saved to models/bigru_attention_twin.pth")
model_path = "models/bigru_attention_twin.pth"
assert os.path.exists(model_path), f"❌ Model not saved at {model_path}"
print(f"✅ Model saved to {model_path}")