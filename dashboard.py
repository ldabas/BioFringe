import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv('./data/viable_dataset.csv', index_col='Date')

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
    forecast = model.predict(data_reshaped)

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
    with st.form("add_new_data"):
        st.markdown("## Add new values")
        col1, col2, col3 = st.columns(3)
        # TODO: Naming of vars is horrendous. We need to change that to something readable
        with col1:
            new_date = st.text_input("Enter **Date**")
            new_ps_q_day = st.number_input("Enter **PS_Q_DAY**", step=1e-4, format="%.4f")
        
        with col2:
            new_tps_q1_day = st.number_input("Enter **TPS_Q1_DAY**", step=1e-4, format="%.4f")
            new_twas_daf_qin_day = st.number_input("Enter **TWAS_DAF_QIN_DAY**", step=1e-4, format="%.4f")
        
        with col3:
            new_dig_s_qout_day = st.number_input("Enter **DIGESTED_SLUDGE_QOUT_DAY**", step=1e-4, format="%.4f")
            new_dig_s_dwtr_ds_after_per_week = st.number_input("Enter **DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK**", step=1e-4, format="%.4f")
        submitted = st.form_submit_button()

        if submitted:
            new_row = np.array([new_ps_q_day, new_tps_q1_day, new_twas_daf_qin_day, new_dig_s_qout_day, 0, new_dig_s_dwtr_ds_after_per_week])

            # Load the model
            model = load_model('./models/lstm_model_multi-io-tomorrow.h5')
            look_back = 30
            past_data = df.iloc[-look_back + 1:].values  # Get the last look_back - 1 days of data

            # Make the prediction
            new_row = forecast(model, new_row.T, past_data)
            append_data(new_row.T.flatten(), new_date)
            st.success("Data added successfully!")

def add_metrics():

    kpi_metric = st.session_state.data[st.session_state.data.index.values == date_filter]
    kpi1, kpi2 = st.columns(2)

    kpi1.metric(
        label="**BIOGAS_PRODUCTION_Q_DAY**",
        value=kpi_metric["BIOGAS_PRODUCTION_Q_DAY"],
    )

    kpi2.metric(
        label="**DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK**",
        value=kpi_metric["DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK"]
    )

def add_figures():
    fig_col1, fig_col2 = st.columns(2)

    with fig_col1:
        st.markdown("## Biogas Production - PS_Q")
        fig = px.density_heatmap(
            data_frame=st.session_state.data, y="DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK", x="BIOGAS_PRODUCTION_Q_DAY"
        )
        st.write(fig)
    
    with fig_col2:
        st.markdown("## Biogas Production - Date")
        fig2 = px.line(data_frame=st.session_state.data, y="BIOGAS_PRODUCTION_Q_DAY", x=st.session_state.data.index.values)
        st.write(fig2)

if __name__ == "__main__":
    st.set_page_config(
        page_title="BioFringe",
        # TODO page_icon: Maybe add BioFringe icon here??
        layout="wide",
    )
    df = get_data()

    # Title
    st.markdown("<h1 style='text-align: center; color: black;'>Biogas production</h1>", unsafe_allow_html=True)

    placeholder = st.empty()

    with placeholder.container():

        if 'data' not in st.session_state:
            st.session_state.data = df

        add_values()

        # Top-level filter
        date_filter = st.selectbox("**Select Date**", pd.unique(st.session_state.data.index.values))
        
        add_metrics()

        add_figures()

        st.markdown("### Detailed Data View")
        st.dataframe(st.session_state.data)
