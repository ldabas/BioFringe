
import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

import Home

def append_data(new_data, new_date):
    st.session_state.data.loc[new_date] = new_data
    st.session_state.data.to_csv('./data/viable_dataset.csv', index=True)

def forecast_next_day(model, past_data, new_row, scaler, look_back):

    # Ensure past_data is 2D and has the correct shape
    assert len(past_data.shape) == 2 and past_data.shape[0] == look_back - 1

    # Ensure new_row is 1D and has the correct shape
    assert len(new_row.shape) == 1 and new_row.shape[0] == past_data.shape[1]

    # Ensure that the biogas production variable is set to zero
    assert new_row[4] == 0

    # Concatenate past_data and new_row along the time dimension
    data = np.concatenate([past_data, new_row[np.newaxis, :]], axis=0)

    # Scale the data
    data = scaler.fit_transform(data)
    data_scaled = scaler.transform(data)

    # Add an extra dimension to match the input shape for the LSTM model
    data_reshaped = np.expand_dims(data_scaled, axis=0)

    # Perform the forecast
    forecast = model(data_reshaped)

    # Invert the scaling
    forecast_inverted = scaler.inverse_transform(forecast[0])

    return forecast_inverted

def forecast(model, new_row, past_data):
    # For now, this will forecast for next day,
    # The functionality could be expanded to forecast for the
    # next 7 days once we add a new model in
    scaler = MinMaxScaler(feature_range=(0, 1))
    look_back = 30
    return forecast_next_day(model, past_data, new_row, scaler, look_back)

def add_values():
    st.markdown("<h1 style='text-align: center; color: black;'>Prediction of biogas production</h1>", unsafe_allow_html=True)
    with st.form("add_new_data"):
        st.markdown("<h1 style='text-align: center; color: black; font-size: 30px;'>Add new values</h1>", unsafe_allow_html=True)
        col_array= st.columns(3)
        # TODO: Naming of vars is horrendous. We need to change that to something readable
        with col_array[0]:
            new_date = st.date_input("Enter **Date**")
            new_ps_q_day = st.number_input("Enter **Primary Sludge (m3/d)**", step=1e-4, format="%.4f")
        
        with col_array[1]:
            new_tps_q1_day = st.number_input("Enter **Thickened Primary Sludge (m3/d)**", step=1e-4, format="%.4f")
            new_twas_daf_qin_day = st.number_input("Enter **Thickened Waste Activated Sludge (m3/d)**", step=1e-4, format="%.4f")
        
        with col_array[2]:
            new_dig_s_qout_day = st.number_input("Enter **Produced Digested Sludge (m3/d)**", step=1e-4, format="%.4f")
            new_dig_s_dwtr_ds_after_per_week = st.number_input("Enter **Digested Dewater Sludge (kg soln)**", step=1e-4, format="%.4f")
        submitted = st.form_submit_button()

        if (hasattr(st.session_state, 'data')):
            look_back = 30
            past_data = st.session_state.data.iloc[-look_back + 1:].values  # Get the last look_back - 1 days of data
        else:
            st.error("You should navigate to the Home Tab initially.")

        if submitted:
            new_row = np.array([new_ps_q_day, new_tps_q1_day, new_twas_daf_qin_day, new_dig_s_qout_day, 0, new_dig_s_dwtr_ds_after_per_week])

            # Load the model
            model = load_model('./models/lstm_model_multi-io-tomorrow.h5')

            # Make the prediction
            new_row = forecast(model, new_row.T, past_data)
            append_data(new_row.T.flatten(), new_date)
            st.success("Data added successfully!")
            biogas = str(new_row.T[4]).replace('[', '')
            biogas = biogas.replace(']','')
            st.write("#### The predicted value of biogas production for " , new_date, " is ", biogas)

if __name__ == "__main__":
    try:
        add_values()
    except:
        pass # This is here so errors are not propagated to the client