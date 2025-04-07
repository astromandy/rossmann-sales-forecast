# main.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import sys


# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('~/rossmann-sales-forecast/src/'), '..')))

from src.etl import carregar_dados, limpar_dados
from src.features import criar_variaveis_temporais, criar_lags, criar_medias_moveis

# ====================
# Paths
# ====================
train_path = "~/rossmann-sales-forecast/data/raw/train.csv"
store_path = "~/rossmann-sales-forecast/data/raw/store.csv"
model_path = "/home/amanda/rossmann-sales-forecast/models/xgb_model.pkl"
fig_path = "/home/amanda/rossmann-sales-forecast/models/evaluation_plot.png"

# ====================
# Load and preprocess data
# ====================
df = carregar_dados(train_path, store_path)
df = limpar_dados(df)
df = criar_variaveis_temporais(df)
df = criar_lags(df, lags=[1, 7, 14])
df = criar_medias_moveis(df, janelas=[7, 14])
df = df.dropna()

# ====================
# Define features and target
# ====================
features = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Month', 'Day', 'Year',
            'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14',
            'Sales_roll_7', 'Sales_roll_14']
target = 'Sales'

X = df[features]
y = df[target]

# ====================
# Model training with TimeSeriesSplit
# ====================
tscv = TimeSeriesSplit(n_splits=3)
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_dist = {
    'n_estimators': np.arange(50, 201, 50),
    'max_depth': np.arange(3, 11),
    'learning_rate': np.linspace(0.01, 0.3, 10),
    'subsample': np.linspace(0.5, 1.0, 6)
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=20,
    cv=tscv,
    scoring='neg_mean_absolute_error',
    random_state=42,
    verbose=1,
    n_jobs=-1
)

search.fit(X, y)
best_model = search.best_estimator_
print("Best hyperparameters found:", search.best_params_)

# ====================
# Final model evaluation
# ====================
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# Fit model on last split
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# ====================
# Save model
# ====================
joblib.dump(best_model, model_path)
print(f"Model saved to {model_path}")

# ====================
# Evaluation plot
# ====================
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:300], label="Actual", color='black')
plt.plot(y_pred[:300], label="Predicted", color='darkorange')
plt.title("Actual vs Predicted Sales (Sample of 300)")
plt.xlabel("Observations")
plt.ylabel("Sales")
plt.legend()
plt.tight_layout()
plt.savefig(fig_path)
plt.close()
print(f"Evaluation plot saved to {fig_path}")

