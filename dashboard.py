import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv('./viable_dataset.csv')

def add_values():
    with st.form("add_new_data"):
        st.markdown("## Add new values")
        col1, col2, col3 = st.columns(3)
        # TODO: Naming of vars is horrendous. We need to change that to something readable
        with col1:
            new_date = st.date_input("Fill in **Date**")
            new_ps_q_day = st.number_input("Fill in **PS_Q_DAY**")
        
        with col2:
            new_tps_q1_day = st.number_input("Fill in **TPS_Q1_DAY**")
            new_twas_daf_qin_day = st.number_input("Fill in **TWAS_DAF_QIN_DAY**")
        
        with col3:
            new_dig_s_qout_day = st.number_input("Fill in **DIGESTED_SLUDGE_QOUT_DAY**")
            new_dig_s_dwtr_ds_after_per_week = st.number_input("Fill in **DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK**")
        submitted = st.form_submit_button()
        if submitted:
            new_entry = {'Date':new_date, 'PS_Q_DAY':new_ps_q_day, 'TPS_Q1_DAY':new_tps_q1_day, 'TWAS_DAF_QIN_DAY':new_twas_daf_qin_day, \
                        'DIGESTED_SLUDGE_QOUT_DAY':new_dig_s_qout_day, 'BIOGAS_PRODUCTION_Q_DAY':0, \
                            'DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK':new_dig_s_dwtr_ds_after_per_week}
            new_entry = pd.Series(new_entry)
            st.session_state.data = pd.concat([st.session_state.data, new_entry.to_frame().T]).reset_index(drop=True)

def add_metrics():

    kpi_metric = st.session_state.data[st.session_state.data["Date"] == date_filter]
    kpi1, kpi2 = st.columns(2)

    kpi1.metric(
        label="BIOGAS_PRODUCTION_Q_DAY",
        value=kpi_metric["BIOGAS_PRODUCTION_Q_DAY"],
    )

    kpi2.metric(
        label="DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK",
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
        fig2 = px.line(data_frame=st.session_state.data, y="BIOGAS_PRODUCTION_Q_DAY", x="Date")
        st.write(fig2)

if __name__ == "__main__":
    st.set_page_config(
        page_title="BioFringe",
        # TODO page_icon: Maybe add BioFringe icon here??
        layout="wide",
    )
    df = get_data()
    # Title
    st.title("Biogas production")

    placeholder = st.empty()

    with placeholder.container():

        if 'data' not in st.session_state:
            st.session_state.data = df

        # Top-level filter
        date_filter = st.selectbox("Select Date", pd.unique(st.session_state.data["Date"]))
        
        add_metrics()

        add_figures()

        add_values()

        st.markdown("### Detailed Data View")
        #st.session_state.data = st.session_state.data.astype(str) # TODO: Weird error about Date ?? Not sure how to resolve
        st.dataframe(st.session_state.data)

            
