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

def append_data(new_data):
    st.session_state.data = pd.concat([st.session_state.data, new_data]).reset_index(drop=True)
    st.session_state.data.to_csv('./data/viable_dataset.csv', index=False)

def forecast_next_day(model, new_row, scaler, look_back):
    # Scale the new row of data
    new_row = scaler.fit_transform(new_row)
    new_row_scaled = scaler.transform(new_row.values.reshape(1, -1))

    # Reshape the data to match the input shape for the LSTM model
    new_row_reshaped = np.reshape(new_row_scaled, (1, look_back, new_row_scaled.shape[1]))

    # Perform the forecast
    forecast = model.predict(new_row_reshaped)

    # Invert the scaling
    forecast_inverted = scaler.inverse_transform(forecast)

    return forecast_inverted

def forecast(model, new_row):
    # For now, this will forecast for next day,
    # The functionality could be expanded to forecast for the
    # next 7 days

    scaler = MinMaxScaler(feature_range=(0, 1))
    look_back = 30
    return forecast_next_day(model, new_row, scaler, look_back)

def add_values():
    with st.form("add_new_data"):
        st.markdown("## Add new values")
        col1, col2, col3 = st.columns(3)
        # TODO: Naming of vars is horrendous. We need to change that to something readable
        with col1:
            new_date = st.text_input("Enter **Date**")
            new_ps_q_day = st.number_input("Enter **PS_Q_DAY**")
        
        with col2:
            new_tps_q1_day = st.number_input("Enter **TPS_Q1_DAY**")
            new_twas_daf_qin_day = st.number_input("Enter **TWAS_DAF_QIN_DAY**")
        
        with col3:
            new_dig_s_qout_day = st.number_input("Enter **DIGESTED_SLUDGE_QOUT_DAY**")
            new_dig_s_dwtr_ds_after_per_week = st.number_input("Enter **DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK**")
        submitted = st.form_submit_button()

        ###
        ### Here we will be using the model to forecast the value of BIOGAS_PRODUCTION_Q_DAY
        ### This should be inside a function called forecast() or something similar
        ###

        if submitted:
            new_entry = {'PS_Q_DAY':new_ps_q_day, 'TPS_Q1_DAY':new_tps_q1_day, 'TWAS_DAF_QIN_DAY':new_twas_daf_qin_day, \
                        'DIGESTED_SLUDGE_QOUT_DAY':new_dig_s_qout_day, 'BIOGAS_PRODUCTION_Q_DAY':0, \
                            'DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK':new_dig_s_dwtr_ds_after_per_week}
            new_entry = pd.DataFrame(new_entry, index=[new_date])
            append_data(new_entry)

            # Load the model
            model = load_model('./models/lstm_model_multi-io-tomorrow.h5')

            temp = forecast(model, new_entry)
            print(temp)
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
        #st.session_state.data = st.session_state.data.astype(str) # TODO: Weird error about Date ?? Not sure how to resolve
        st.dataframe(st.session_state.data)
