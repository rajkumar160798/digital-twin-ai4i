import pandas as pd
import numpy as np
import torch
from twin_model.model import GRUDigitalTwin
from twin_model.utils import load_model
from simulator.synthetic_data_tools import inject_spike
from simulator.deviation_score import compute_deviation

# Load normalized healthy data
df = pd.read_csv("data/processed/ai4i_cleaned.csv")
sensor_cols = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
df_healthy = df[df['Machine failure'] == 0].reset_index(drop=True)

# Extract a sample sequence
sequence = df_healthy[sensor_cols].iloc[100:120].values

# Load GRU model
model = load_model(GRUDigitalTwin, "models/gru_digital_twin.pth", len(sensor_cols))

# Inject a torque spike
faulty_seq = sequence.copy()
faulty_seq[:, 3] = inject_spike(pd.Series(faulty_seq[:, 3]), location=15, magnitude=0.8).values


# Compute deviation
deviation = compute_deviation(model, sequence, faulty_seq)
print("Deviation score per timestep:", deviation)
print("Mean deviation score:", np.mean(deviation))
# Save the faulty sequence for further analysis
faulty_df = pd.DataFrame(faulty_seq, columns=sensor_cols)
faulty_df.to_csv("data/faulty_sequence.csv", index=False)

import matplotlib.pyplot as plt

# Generate predictions for the healthy sequence
with torch.no_grad():
	input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)  # shape: (1, seq_len, num_features)
	pred = model(input_tensor).squeeze(0).numpy()  # shape: (seq_len, num_features)

plt.figure(figsize=(10, 4))
plt.plot(sequence[:, 3], label="Healthy Torque", linewidth=2)
plt.plot(faulty_seq[:, 3], label="Faulty Torque", linewidth=2)
plt.plot(pred[:, 3], label="GRU Predicted Torque", linestyle='--', linewidth=2)
plt.legend()
plt.title("Torque Signal Comparison (Healthy vs Faulty vs Predicted)")
plt.xlabel("Timestep")
plt.ylabel("Torque (Normalized)")
plt.grid(True)
plt.tight_layout()
plt.show()
# Save the plot
plt.savefig("../outputs/reports/torque_signal_comparison.png")
