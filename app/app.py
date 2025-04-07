# app/app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('/home/amanda/rossmann-sales-forecast/src/'), '..')))

from src.etl import carregar_dados, limpar_dados
from src.features import criar_variaveis_temporais, criar_lags, criar_medias_moveis
from src.predict import prever_proximos_dias

# File paths
train_path = "~/rossmann-sales-forecast/data/raw/train.csv"
store_path = "~/rossmann-sales-forecast/data/raw/store.csv"
model_path = "/home/amanda/rossmann-sales-forecast/models/xgb_model.pkl"
prediction_path = "/home/amanda/rossmann-sales-forecast/data/predictions/"


# Carregar o modelo
MODEL_PATH = "/home/amanda/rossmann-sales-forecast/models/xgb_model.pkl"
model = joblib.load(MODEL_PATH)

# Configuração do app
st.set_page_config(page_title="Rossmann Sales Forecast", layout="wide")

st.title("📊 Rossmann Sales Forecasting App")
st.markdown("Previsão de vendas futuras com modelo XGBoost.")

# Sidebar
st.sidebar.header("Previsão personalizada")
store_id = st.sidebar.number_input("Store ID", min_value=1, max_value=1115, value=1, step=1)
n_dias = st.sidebar.slider("Dias futuros a prever", min_value=1, max_value=30, value=7)

# Carregar dados
data_train_path = "/home/amanda/rossmann-sales-forecast/data/raw/train.csv"
data_store_path = "/home/amanda/rossmann-sales-forecast/data/raw/store.csv"

df = carregar_dados(data_train_path, data_store_path)
df = limpar_dados(df)

# Previsão
df_previsto = prever_proximos_dias(df, model, store_id=store_id, n_dias=n_dias)

if isinstance(df_previsto, str):
    st.warning(df_previsto)
else:
    st.subheader(f"📈 Previsão para a loja {store_id}")
    st.dataframe(df_previsto)

    # Plot
    st.subheader("Visualização")
    historico = df[df['Store'] == store_id].sort_values('Date').copy()
    historico = historico[historico['Sales'] > 0]
    historico_recente = historico.tail(30)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(historico_recente['Date'], historico_recente['Sales'], label="Histórico", color="blue", marker='o')
    ax.plot(df_previsto['Data_prevista'], df_previsto['Previsao_vendas'], label="Previsão", color="orange", marker='o')
    ax.set_title(f"Histórico e previsão de vendas - Loja {store_id}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Vendas")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

