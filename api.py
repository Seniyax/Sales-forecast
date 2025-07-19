from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from typing import List, Dict
from SalesFeatureEngineer import FeatureEngineer
app = FastAPI()
model = joblib.load('best_xgboost.pkl')
feature_engineer = FeatureEngineer()


# Load initial history (run once at startup)
@app.on_event("startup")
def load_history():
    try:
        history = pd.read_csv('sales_history.csv', parse_dates=['date'])
        feature_engineer.add_new_data(history)
    except FileNotFoundError:
        print("No history found - starting fresh")


@app.post("/predict")
async def predict(sales_data: List[Dict]):
    """
    Expects format: [{"date": "2023-01-01", "store": 1, "item": 1, "sales": 100}, ...]
    """
    try:
        # Convert and validate input
        new_data = pd.DataFrame(sales_data)
        new_data['date'] = pd.to_datetime(new_data['date'])

        # Add data and generate features
        engineered_data = feature_engineer.add_new_data(new_data)

        # Prepare for prediction (drop non-feature columns)
        predict_data = engineered_data.drop(columns=['date', 'store', 'item', 'sales'])

        # Make predictions
        predictions = model.predict(predict_data)

        # Save updated history
        feature_engineer.history.to_csv('sales_history.csv', index=False)

        return {
            "predictions": predictions.tolist(),
            "features": engineered_data.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))