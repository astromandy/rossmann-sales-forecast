import sqlite3
import pandas as pd

# Função para criar banco de dados e tabelas
def create_db(db_name="rossmann.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Criar tabela de stores
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stores (
        store INTEGER PRIMARY KEY,
        name TEXT,
        type TEXT,
        region TEXT
    );
    """)

    # Criar tabela de vendas
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sales (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        store INTEGER,
        date DATE,
        sales INTEGER,
        promo INTEGER,
        state_holiday TEXT,
        FOREIGN KEY(store) REFERENCES stores(store)
    );
    """)

    conn.commit()
    conn.close()

# Chamar a função para criar o DB
create_db()

