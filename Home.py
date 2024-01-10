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
    try:
        st.set_page_config(
            page_title="BioFringe",
            # TODO page_icon: Maybe add BioFringe icon here??
            layout="wide",
        )
        df = get_data()

        # Title
        st.markdown("<h1 style='text-align: center; color: black;'>BioFringe</h1>", unsafe_allow_html=True)

        placeholder = st.empty()

        with placeholder.container():

            if 'data' not in st.session_state:
                st.session_state.data = df

            # Top-level filter
            date_filter = st.selectbox("**Select Date**", pd.unique(st.session_state.data.index.values))
            
            add_metrics()

            add_figures()

            st.markdown("### Detailed Data View")
            st.dataframe(st.session_state.data)
    except:
        pass # This is here so errors are not propagated to the client
