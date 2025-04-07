# shap_analysis/shap_interpretation.py

import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('/home/amanda/rossmann-sales-forecast/src/'), '..')))

from src.etl import carregar_dados, limpar_dados
from src.features import criar_variaveis_temporais, criar_lags, criar_medias_moveis

# Paths

train_path = "~/rossmann-sales-forecast/data/raw/train.csv"
store_path = "~/rossmann-sales-forecast/data/raw/store.csv"
model_path = "/home/amanda/rossmann-sales-forecast/models/xgb_model.pkl"
output_dir = "/home/amanda/rossmann-sales-forecast/shap_analysis/"

os.makedirs(output_dir, exist_ok=True)

# Load model
model = joblib.load(model_path)

# Load and preprocess data
df = carregar_dados(train_path, store_path)
df = limpar_dados(df)
df = criar_variaveis_temporais(df)
df = criar_lags(df, lags=[1, 7, 14])
df = criar_medias_moveis(df, janelas=[7, 14])
df = df.dropna()

# Select features and target
features = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Month', 'Day', 'Year',
            'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14',
            'Sales_roll_7', 'Sales_roll_14']
X = df[features]

# Explain with SHAP
explainer = shap.Explainer(model)
shap_values = explainer(X)

# Summary plot (bar)
plt.figure()
shap.plots.bar(shap_values, show=False)
plt.title("SHAP Feature Importance (Bar)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_feature_importance_bar.png"))
plt.close()

# Summary plot (dot)
plt.figure()
shap.plots.beeswarm(shap_values, show=False)
plt.title("SHAP Feature Importance (Beeswarm)")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_feature_importance_beeswarm.png"))
plt.close()

# Force plot for a single prediction (first row)
force_plot_path = os.path.join(output_dir, "shap_force_plot.html")
shap.save_html(force_plot_path, shap.plots.force(shap_values[0]))

print(f"SHAP analysis completed. Files saved in: {output_dir}")

