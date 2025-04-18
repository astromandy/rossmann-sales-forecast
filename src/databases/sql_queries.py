# Function to fetch query results from the database
import sqlite3
import pandas as pd

def fetch_query_results(query, db_name="rossmann.db"):
    """
    Executes an SQL query on the database and returns the result as a DataFrame.
    
    :param query: The SQL query to be executed.
    :param db_name: The name of the database (default is 'rossmann.db').
    :return: DataFrame containing the query results.
    """
    # Connect to the SQLite database
    conn = sqlite3.connect(db_name)
    
    # Use pandas to execute the query and return the results as a DataFrame
    df = pd.read_sql(query, conn)
    
    # Close the database connection
    conn.close()
    
    return df

# SQL Queries for specific analysis

# Example: Average sales per store
def avg_sales_per_store():
    """
    Query to fetch the average sales per store.
    :return: SQL query string for average sales per store.
    """
    query = """
    SELECT store, AVG(sales) as avg_sales
    FROM sales
    GROUP BY store;
    """
    return query

# Example: Sales by promotion and store
def sales_by_promo_and_store():
    """
    Query to fetch total sales by promotion and store.
    :return: SQL query string for sales by promotion and store.
    """
    query = """
    SELECT store, promo, SUM(sales) as total_sales
    FROM sales
    GROUP BY store, promo;
    """
    return query

# Example: Sales by state holiday
def sales_by_state_holiday():
    """
    Query to fetch total sales by state holiday.
    :return: SQL query string for sales by state holiday.
    """
    query = """
    SELECT state_holiday, SUM(sales) as total_sales
    FROM sales
    GROUP BY state_holiday;
    """
    return query

# Example: Daily sales per store
def daily_sales_per_store():
    """
    Query to fetch daily sales for each store.
    :return: SQL query string for daily sales per store.
    """
    query = """
    SELECT store, date, sales
    FROM sales
    ORDER BY date;
    """
    return query
