import sqlite3
import pandas as pd

# Function to create the database and tables
def create_db(db_name="rossmann.db"):
    # Connect to (or create) the SQLite database
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Create 'stores' table if it doesn't already exist
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stores (
        store INTEGER PRIMARY KEY,
        name TEXT,
        type TEXT,
        region TEXT
    );
    """)

    # Create 'sales' table if it doesn't already exist
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

    # Commit changes and close the connection
    conn.commit()
    conn.close()

# Call the function to create the database
create_db()

