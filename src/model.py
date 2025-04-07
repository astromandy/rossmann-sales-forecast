import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('/home/amanda/rossmann-sales-forecast/src/'), '..')))

from src.etl import carregar_dados, limpar_dados
from src.features import criar_variaveis_temporais, criar_lags, criar_medias_moveis

# === 1. Load and prepare data ===
df = carregar_dados(
    "/home/amanda/rossmann-sales-forecast/data/raw/train.csv",
    "/home/amanda/rossmann-sales-forecast/data/raw/store.csv"
)

df = limpar_dados(df)
df = criar_variaveis_temporais(df)
df = criar_lags(df, lags=[1, 7, 14])
df = criar_medias_moveis(df, janelas=[7, 14])
df = df.dropna()

# === 2. Define features and target ===
features = [
    'Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Month', 'Day', 'Year',
    'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14',
    'Sales_roll_7', 'Sales_roll_14'
]
target = 'Sales'

X = df[features]
y = df[target]

# === 3. Validation strategy ===
tscv = TimeSeriesSplit(n_splits=3)

# === 4. Define base model ===
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# === 5. Hyperparameter tuning ===
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
print("Best parameters:", search.best_params_)

# === 6. Evaluate on last fold ===
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# === 7. Feature importance ===
xgb.plot_importance(best_model, max_num_features=10)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# === 8. Predicted vs Real Sales ===
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:500], label='Actual')
plt.plot(y_pred[:500], label='Predicted')
plt.title("Actual vs Predicted Sales (Sample of 500)")
plt.legend()
plt.tight_layout()
plt.show()

# === 9. Save model ===
import joblib
joblib.dump(best_model, "/home/amanda/rossmann-sales-forecast/models/xgb_model.pkl")
print("Model saved to models/xgb_model.pkl")

