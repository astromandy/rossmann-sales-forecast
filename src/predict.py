# src/predict.py

import pandas as pd
import numpy as np
from datetime import timedelta
import sys 
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('/home/amanda/rossmann-sales-forecast/src/'), '..')))

from src.etl import carregar_dados, limpar_dados
from src.features import criar_variaveis_temporais, criar_lags, criar_medias_moveis


def prever_proximos_dias(df, model, store_id, n_dias=7):
    # Filtra os dados da loja
    df_loja = df[df['Store'] == store_id].sort_values('Date').copy()

    # Verifica se há dados suficientes
    if df_loja[df_loja['Sales'] > 0].shape[0] < 14:
        return f"A loja {store_id} não possui pelo menos 14 dias com vendas > 0 para gerar previsão."

    # Cria cópia dos últimos dados para janelamento
    df_prev = df_loja.copy()
    ult_data = df_prev['Date'].max()

    previsoes = []

    for i in range(n_dias):
        nova_data = ult_data + timedelta(days=1)
        nova_linha = {
            'Store': store_id,
            'Date': nova_data,
            'Sales': np.nan,
            'Customers': np.nan,
            'Open': 1,
            'Promo': 0,
            'StateHoliday': 0,
            'SchoolHoliday': 0
        }

        df_prev = pd.concat([df_prev, pd.DataFrame([nova_linha])], ignore_index=True)
        df_prev = criar_variaveis_temporais(df_prev)
        df_prev = criar_lags(df_prev, lags=[1, 7, 14])
        df_prev = criar_medias_moveis(df_prev, janelas=[7, 14])

        linha_prev = df_prev[df_prev['Date'] == nova_data].copy()
        features = ['Store', 'DayOfWeek', 'Promo', 'SchoolHoliday', 'Month', 'Day', 'Year',
                    'Sales_lag_1', 'Sales_lag_7', 'Sales_lag_14', 'Sales_roll_7', 'Sales_roll_14']

        if linha_prev[features].isnull().values.any():
            break

        pred = model.predict(linha_prev[features])[0]
        previsoes.append({'Store': store_id, 'Data_prevista': nova_data, 'Previsao_vendas': pred})

        df_prev.loc[df_prev['Date'] == nova_data, 'Sales'] = pred
        ult_data = nova_data

    return pd.DataFrame(previsoes)

