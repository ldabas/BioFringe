import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

class BiogasOptimizer:
    def __init__(self, data_path):
        # Load data
        self.data = pd.read_csv(data_path)
        self.data = self.data.drop(columns=['Date'])

        # Split data
        self.y = self.data['BIOGAS_PRODUCTION_Q_DAY']
        self.X = self.data.drop(columns=['BIOGAS_PRODUCTION_Q_DAY'])

        # Scale data
        self.scaler = MinMaxScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        # Train model
        self.model = GradientBoostingRegressor(random_state=42)
        self.model.fit(self.X_scaled, self.y)

    def optimize(self, min_values=None, max_values=None):
        # Use min and max values from data if not provided
        if min_values is None:
            min_values = self.X.min().to_dict()
        if max_values is None:
            max_values = self.X.max().to_dict()

        # Convert min/max values to arrays in the correct order
        min_values = np.array([min_values[col] for col in self.X.columns])
        max_values = np.array([max_values[col] for col in self.X.columns])

        # Define objective function
        def objective(input_features, model, scaler, min_values, max_values):
            input_features = np.clip(input_features, min_values, max_values)
            input_features = input_features.reshape(1, -1)
            input_features_scaled = scaler.transform(input_features)
            predicted_output = model.predict(input_features_scaled)
            return -predicted_output

        # Initial guess for the input features
        initial_guess = self.X.mean().values

        # Bounds for the input features
        bounds = [(min_values[i], max_values[i]) for i in range(len(min_values))]

        # Perform the optimization
        result = minimize(objective, initial_guess, args=(self.model, self.scaler, min_values, max_values), bounds=bounds, method='L-BFGS-B')

        # The optimal input features
        optimal_input_features = result.x

        return optimal_input_features
    
    def get_optimized(self):
        st.markdown("<h1 style='text-align: center; color: black;'>Optimization of variables</h1>", unsafe_allow_html=True)
        with st.form('max_min_values'):
            st.markdown("## Add minimum and maximum values")
            min = st.text_input("Enter minimum values separated by comma", help="i.e. 100,200,300,400,500")
            max = st.text_input("Enter maximum values separated by comma", help="i.e. 100,200,300,400,500")
            submitted = st.form_submit_button()
            if submitted:
                min = min.split(',')
                max = max.split(',')
                min = [int(input) for input in min]
                max = [int(input) for input in max]
                min = dict(zip(self.X.columns, min))
                max = dict(zip(self.X.columns, max))
                result = self.optimize(min_values=min, max_values=max)
                st.write("#### Your optimized values are")
                st.write("##### PS_Q_DAY: ", str(result[0]))
                st.write("##### TPS_Q1_DAY: ", str(result[1]))
                st.write("##### TWAS_DAF_QIN_DAY: ", str(result[2]))
                st.write("##### DIGESTED_SLUDGE_QOUT_DAY: ", str(result[3]))
                st.write("##### DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK: ", str(result[4]))


if __name__ == "__main__":
    optimizer = BiogasOptimizer('./data/viable_dataset.csv')
    optimizer.get_optimized()