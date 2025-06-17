# digital-twin-ai4i
Digital Twin Module for ProvansIQ using the AI4I 2020 dataset

## Project Structure
```text
digital-twin-ai4i/
├── data/
│   ├── raw/                         # Original dataset (ai4i2020.csv)
│   ├── processed/                   # Cleaned, normalized time-series data
│   └── synthetic/                   # Failure-injected synthetic datasets
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Initial EDA, visualization
│   ├── 02_digital_twin_model.ipynb  # Forecast model (GRU/LSTM/Prophet)
│   ├── 03_synthetic_failure_gen.ipynb # Inject and label synthetic anomalies
│   ├── 04_failure_classifier.ipynb  # Train failure prediction model
│   └── 05_shap_explanations.ipynb   # SHAP visualizations for root cause
│
├── twin_model/
│   ├── __init__.py
│   ├── model.py                     # GRU/Prophet model logic
│   ├── train.py                     # Training and evaluation functions
│   └── utils.py                     # Preprocessing, plotting, normalization
│
├── simulator/
│   ├── scenario_runner.py           # Run what-if scenarios
│   ├── deviation_score.py           # Measure twin vs real behavior
│   └── synthetic_data_tools.py      # Inject faults, add drift
│
├── app/
│   ├── streamlit_app.py             # UI for dashboard (optional)
│   └── plots/                       # Saved charts and overlays
│
├── models/
│   ├── twin_model.pkl               # Saved model weights
│   ├── classifier_model.pkl         # Failure prediction model
│   └── scaler.pkl                   # Normalization scaler
│
├── outputs/
│   ├── reports/                     # Model summary, metrics
│   └── visualizations/              # SHAP, twin overlays
│
├── requirements.txt
├── README.md
└── digital_twin_architecture.png    # System diagram (for blog/paper)
```

