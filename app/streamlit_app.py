import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import twin_model
from twin_model.model import GRUDigitalTwin
from twin_model.utils import load_model
from simulator.synthetic_data_tools import inject_spike, inject_gradual_drift
from simulator.deviation_score import compute_deviation

# ----- CONFIG -----
st.set_page_config(page_title="Digital Twin Failure Simulator", layout="wide")
sensor_cols = ['Air temperature [K]', 'Process temperature [K]', 
               'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

# ----- LOAD DATA + MODEL -----
@st.cache_data
def load_clean_data():
    df = pd.read_csv("data/processed/ai4i_cleaned.csv")
    return df[df['Machine failure'] == 0].reset_index(drop=True)

@st.cache_resource
def load_gru_model():
    return load_model(GRUDigitalTwin, "models/gru_digital_twin.pth", len(sensor_cols))

df_healthy = load_clean_data()
model = load_gru_model()

# ----- SIDEBAR -----
st.sidebar.title("⚙️ Simulator Options")
failure_type = st.sidebar.selectbox("Inject Fault Type", ["None", "Torque Spike", "Overheating", "Wearout"])
start_idx = st.sidebar.slider("Sample Window Start Index", 0, len(df_healthy) - 20, 100)

# ----- SIMULATION -----
sequence = df_healthy[sensor_cols].iloc[start_idx:start_idx+20].values
faulty_seq = sequence.copy()

if failure_type == "Torque Spike":
    faulty_seq[:, 3] = inject_spike(faulty_seq[:, 3], location=15, magnitude=0.8)
elif failure_type == "Overheating":
    faulty_seq[:, 1] = inject_gradual_drift(faulty_seq[:, 1], strength=0.4)
elif failure_type == "Wearout":
    faulty_seq[:, 4] = inject_gradual_drift(faulty_seq[:, 4], strength=0.3)

# ----- GRU PREDICTION -----
model.eval()
with torch.no_grad():
    pred = model(torch.tensor(sequence).unsqueeze(0).float())[0].numpy()

# ----- DEVIATION SCORE -----
deviation = compute_deviation(model, sequence, faulty_seq)
mean_deviation = float(np.mean(deviation))

# ----- UI: VISUALS -----
st.title("Digital Twin Simulator Dashboard")
st.subheader(f"Fault Type: **{failure_type}** | Mean Deviation: **{mean_deviation:.4f}**")

# Plot actual vs predicted vs faulty
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(sequence[:, 3], label="Healthy Torque", linewidth=2)
ax1.plot(faulty_seq[:, 3], label="Faulty Torque", linewidth=2)
ax1.plot(pred[:, 3], label="GRU Predicted Torque", linestyle='--', linewidth=2)
ax1.set_title("Torque Over Time (Healthy vs Faulty vs Predicted)")
ax1.set_xlabel("Timestep")
ax1.set_ylabel("Torque (normalized)")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# Deviation heatmap
st.subheader("Deviation Heatmap (Twin vs Faulty)")
fig2, ax2 = plt.subplots(figsize=(10, 1.5))
sns.heatmap([deviation], cmap="Reds", cbar=True, xticklabels=range(20), yticklabels=["Deviation"], ax=ax2)
ax2.set_xlabel("Timestep")
st.pyplot(fig2)

# Alert
if mean_deviation > 0.2:
    st.error("High deviation detected! Possible failure behavior.")
else:
    st.success("Deviation within normal range. Machine operating normally.")

# ----- FOOTER -----
st.sidebar.markdown("---")
st.sidebar.markdown("© 2025 Digital Twin Simulator")
st.sidebar.markdown("Built with Streamlit and PyTorch")
# Add a link to the GitHub repository
st.sidebar.markdown("[GitHub Repository](https://github.com/rajkumar160798/digital-twin-ai4i)")
# created by Raj kumar myakala 
st.sidebar.markdown("Created by Raj Kumar Myakala")
st.sidebar.markdown("[Contact](mailto:myakalarajkumar1998@gmail.com)")
st.sidebar.markdown(f"[LinkedIn](https://www.linkedin.com/in/raj-kumar-myakala-927860264/)")
