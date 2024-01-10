import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize

st.set_page_config(layout="wide")

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
            st.markdown("<h1 style='text-align: center; color: black; font-size: 30px;'> Add minimum and maximum values</h1>", unsafe_allow_html=True)
            col_array = st.columns(5, gap="small")
            var_array = ["PS_Q_DAY", "TPS_Q1_DAY", "TWAS_DAF_QIN_DAY", "DIGESTED_SLUDGE_QOUT_DAY", "DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK"]
            min = []
            max = []
            for i in range(len(col_array)):
                with col_array[i]:
                    min.append(st.number_input("Enter **{}** min".format(var_array[i]), step=1e-4, format="%.4f"))
                    max.append(st.number_input("Enter **{}** max".format(var_array[i]), step=1e-4, format="%.4f"))

            submitted = st.form_submit_button()
            if submitted:
                min = dict(zip(self.X.columns, min))
                max = dict(zip(self.X.columns, max))
                result = self.optimize(min_values=min, max_values=max)
                pred = pd.DataFrame.from_dict({var_array[0]: [str(result[0])], \
                        var_array[1]: [str(result[1])], \
                        var_array[2]: [str(result[2])], \
                        var_array[3]: [str(result[3])], \
                        var_array[4]: [str(result[4])]})
                st.markdown("<h1 style='text-align: center; color: black; font-size: 30px;'>Optimized Values</h1>", unsafe_allow_html=True)
                st.dataframe(pred, hide_index=True)

if __name__ == "__main__":
    try:
        optimizer = BiogasOptimizer('./data/viable_dataset.csv')
        optimizer.get_optimized()
    except:
        pass # This is here so errors are not propagated to the client