# 📊 Rossmann Sales Forecast

This project builds a machine learning model to forecast sales for Rossmann stores using historical sales data and store information.

It includes:
- Data preprocessing (ETL) and database setup (SQLite)
- Feature engineering (lags, rolling averages, calendar features)
- Model training with time-series cross-validation
- Prediction of future sales
- Streamlit application for user interaction
- SHAP analysis for explainability
- SQL queries for data exploration and reporting

## 📁 Project Structure

```
rossmann-sales-forecast/
│
├── data/
│   ├── raw/                 # Raw data (train.csv, store.csv)
│   ├── processed/           # Cleaned and feature-enhanced data
│   └── predictions/         # Generated predictions
│
├── models/
│   └── xgb_model.pkl        # Trained XGBoost model
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory Data Analysis (EDA) with visual insights
│   ├── 02_etl_features.ipynb    # ETL process and feature engineering (lags, rolling means)
│   ├── 03_model_eval.ipynb      # Model training, hyperparameter tuning, and evaluation
│   ├── 04_future_predictions.ipynb # Predict future sales using trained model
│   └── 05_reports.ipynb         # Automated reports and business insights visualization
│   └── 06_sql_queries.ipynb     # SQL queries for advanced analysis
|
├── src/
│   ├── __init__.py
    ├── databases/           # Directory for database-related code
│   │   ├── __init__.py
│   │   ├── create_db.py     # Code for creating the database and tables
│   │   ├── populate_db.py   # Code for populating the database with data
│   │   └── sql_queries.py  # SQL query strings and query execution functions
│   ├── etl.py               # Data loading and cleaning functions
│   ├── features.py          # Feature engineering functions
│   ├── model.py             # Model training and saving logic
│   └── predict.py           # Future predictions
│
├── app/
│   └── app.py               # Streamlit application
│
├── shap_analysis/
│   └── shap_interpretation.py  # SHAP analysis and plots
│
├── main.py                 # Runs the full training & prediction pipeline
├── README.md               # Project overview and instructions
├── requirements.txt        # Required Python packages
└── .gitignore              # Files and folders to ignore in version control
```

## 🖥️ How to Run

To get started with the Rossmann Sales Forecast project:

1. **Clone the repository** and navigate into the project directory:
   git clone https://github.com/astromandy/rossmann-sales-forecast.git
   cd rossmann-sales-forecast


2. **Install dependencies** listed in `requirements.txt`:

   pip install -r requirements.txt

3. **Set up the database by running**

    python src/create_db.py 
    
4. **Populate the database by running**

    python src/populate_db.py
    
3. **Run the complete machine learning pipeline**:

   python main.py


4. *(Optional)* **Launch the interactive Streamlit app** for store-level sales forecasting:

   streamlit run app/app.py
   
