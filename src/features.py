def criar_variaveis_temporais(df):
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    return df

def criar_lags(df, col='Sales', lags=[1, 7, 14]):
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df.groupby('Store')[col].shift(lag)
    return df

def criar_medias_moveis(df, col='Sales', janelas=[7, 14]):
    for janela in janelas:
        df[f'{col}_roll_{janela}'] = df.groupby('Store')[col].shift(1).rolling(window=janela).mean()
    return df

