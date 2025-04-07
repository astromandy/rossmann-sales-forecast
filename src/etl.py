import pandas as pd

def carregar_dados(caminho_treino, caminho_lojas):
    df_train = pd.read_csv(caminho_treino)
    df_store = pd.read_csv(caminho_lojas)
    df = pd.merge(df_train, df_store, how='left', on='Store')

    # Converter coluna de data
    df['Date'] = pd.to_datetime(df['Date'])
    return df

def limpar_dados(df):
    # Remove lojas fechadas ou com vendas zero (opcional)
    df = df[df['Open'] != 0]
    df = df[df['Sales'] > 0]

    # Remove colunas que não serão usadas agora
    if 'Customers' in df.columns:
        df = df.drop(columns=['Customers'])
    if 'Open' in df.columns:
        df = df.drop(columns=['Open'])

    return df

