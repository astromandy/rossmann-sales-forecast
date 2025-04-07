import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('/home/amanda/rossmann-sales-forecast/src/'), '..')))

from src.etl import carregar_dados, limpar_dados
from src.features import criar_variaveis_temporais, criar_lags, criar_medias_moveis

# 1. Paths for data and model
train_path = "/home/amanda/rossmann-sales-forecast/data/raw/train.csv"
store_path = "/home/amanda/rossmann-sales-forecast/data/raw/store.csv"
model_output_path = "/home/amanda/rossmann-sales-forecast/models/xgb_model.pkl"

# 2. Load and prepare the data
df = carregar_dados(train_path, store_path)
df = limpar_dados(df)
df = criar_variaveis_temporais(df)
df = criar_lags(df, lags=[1, 7, 14])
df = criar_medias_moveis(df, janelas=[7, 14])
df = df.dropna()

# 3. Define features and target
features = [
    'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Month', 'Day', 'Year',
    'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14',
    'Sales_roll_7', 'Sales_roll_14'
]
target = 'Sales'

X = df[features]
y = df[target]

# 4. Time series cross-validation
tscv = TimeSeriesSplit(n_splits=3)

# 5. Model with previously optimized parameters
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42
)

# 6. Train and evaluate on the last fold
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# 7. Save model
os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
joblib.dump(model, model_output_path)
print(f"✅ Model saved to: {model_output_path}")

# 8. Plot feature importance
xgb.plot_importance(model, max_num_features=10)
plt.title("Feature Importance")
plt.tight_layout()
plt.savefig("/home/amanda/rossmann-sales-forecast/models/feature_importance.png")
plt.close()

