import streamlit as st
import pandas as pd
import numpy as np
from  SalesFeatureEngineer import FeatureEngineer
import joblib
import os


if not os.path.exists('best_xgboost.pkl'):
    st.error("Model file 'best_xgboost.pkl' not found!")
    st.stop()


try:
    model = joblib.load('best_xgboost.pkl')
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

engineer = FeatureEngineer()

st.title("Store Sales Forecasting")


uploaded_file = st.file_uploader("Upload new sales data (CSV)")
if uploaded_file:
    new_data = pd.read_csv(uploaded_file, parse_dates=['date'])
    with st.spinner('Processing...'):
        # Generate features
        processed = engineer.add_new_data(new_data)

       
        features = processed.drop(columns=['date'])
        processed['prediction'] = model.predict(features)

   
        st.dataframe(processed)

        # Plot
        selected_store = st.selectbox("Select store", processed['store'].unique())
        selected_item = st.selectbox("Select item", processed['item'].unique())

        plot_data = processed[
            (processed['store'] == selected_store) &
            (processed['item'] == selected_item)
            ].set_index('date')

        st.line_chart(plot_data[['prediction']])
