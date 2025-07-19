# Sales Forecasting System
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)

An end-to-end machine learning system for predicting store-item sales using XGBoost with feature engineering and local deployment options.

## Features
- Temporal feature engineering (lags, rolling statistics)
- XGBoost model training with Optuna hyperparameter tuning
- Local deployment via:
  - FastAPI REST endpoint
  - Streamlit interactive dashboard
- Automated feature generation for new predictions

## Project Structure
```` bash
sales-forecast/
├── data/                   
│     └── train.csv         # training dataset
├── notebooks
│      └──Ex_1.ipynb        # experimentations  
│      └──notebooks_1.ipynb # experimentations
│      └──xgboost.ipynb     # training 
├── best_xgboost.pkl        # Saved model  
├── api.py                  # FastAPI implementation
├── dashboard.py            # Streamlit dashboard
└── SalesFeatureEngineer.py # Feature generation logic
````
## Requirements
- Python 3.9+
- XGBoost 1.7+
- Pandas 2.0+
- Streamlit 1.28+
- FastAPI 0.95+
## Installation
1.Clone the Repository:
````bash
git clone https://github.com/Seniyax/Sales-forecast.git
cd sales-forecast
````
2.Install the Requirements
3.Run the api
````bash
uvicorn api:app --reload
````
4.Run the streamlit for local inference
````bash
streamlit run dashboard.py
````
## Dataset
The model was trained on kaggle's Store Item Demand Forecasting Challenge dataset
## Future Improvements
1.Real-Time inference

2.Implement model monitor to  data drift detection



<img width="1296" height="671" alt="Screenshot 2025-07-19 170815" src="https://github.com/user-attachments/assets/937697ad-d0a0-4e89-973b-785304279e14" />
