import pandas as pd
import numpy as np
from datetime import datetime


class FeatureEngineer:
    def __init__(self,min_history=30):
        self.min_history = min_history
        self.history = None
        self.lastValues = {}




    def add_new_data(self, new_rows):

            # Convert new data to DataFrame if single row
            if isinstance(new_rows, dict):
                new_rows = pd.DataFrame([new_rows])

            # Combine with history
            if self.history is None:
                self.history = new_rows.copy()
            else:
                self.history = pd.concat([self.history, new_rows]).sort_values(['store', 'item', 'date'])

            # Generate features for all data
            self._generate_features()

            # Return only the newest rows with features
            return self.history.tail(len(new_rows))

    def _generate_features(self):
            """Internal method to create all features"""
            # Temporal features
            self.history['year'] = self.history['date'].dt.year
            self.history['month'] = self.history['date'].dt.month
            self.history['dow'] = self.history['date'].dt.dayofweek
            self.history['is_weekend'] = self.history['dow'].isin([5, 6]).astype(int)

            # Grouped operations
            grouped = self.history.groupby(['store', 'item'])

            # Lag features
            #self.history['lag_1'] = grouped['sales'].shift(1)
            #self.history['lag_7'] = grouped['sales'].shift(7)
            #self.history['lag_30'] = grouped['sales'].shift(30)

            # Rolling features
            #self.history['rolling_mean_7'] = grouped['sales'].shift(1).rolling(7).mean()
            #self.history['rolling_std_7'] = grouped['sales'].shift(1).rolling(7).std()
            #self.history['sales_dif'] = grouped['sales'].diff()

            # Update last known values
            for (store, item), group in grouped:
                key = f"{store}_{item}"
                self.lastValues[key] = {
                    'lag_1': group['sales'].iloc[-1],
                    'lag_7': group['sales'].iloc[-7] if len(group) >= 7 else np.nan,
                    'lag_30': group['sales'].iloc[-30] if len(group) >= 30 else np.nan,
                    'last_date': group['date'].iloc[-1]
                }

            # Remove rows with missing features
            self.history = self.history.dropna(subset=['lag_1', 'lag_7', 'rolling_mean_7'])