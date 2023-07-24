import time
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv('./viable_dataset.csv')

def append_data(df, new_data):
    new_row = pd.DataFrame(new_data, index=[0])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv('./viable_dataset.csv', index=False)

if __name__ == "__main__":
    st.set_page_config(
        page_title="BioFringe",
        # TODO page_icon: Maybe add BioFringe icon here??
        layout="wide",
    )
    df = get_data()
    # Title
    st.title("Biogas production")

    # Top-level filter
    date_filter = st.selectbox("Select Date", pd.unique(df["Date"]))
    kpi_metric = df[df["Date"] == date_filter]

    placeholder = st.empty()

    with placeholder.container():
        kpi1, kpi2 = st.columns(2)

        kpi1.metric(
            label="BIOGAS_PRODUCTION_Q_DAY",
            value=kpi_metric["BIOGAS_PRODUCTION_Q_DAY"],
        )

        kpi2.metric(
            label="DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK",
            value=kpi_metric["DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK"]
        )

        fig_col1, fig_col2 = st.columns(2)

        with fig_col1:
            st.markdown("## Biogas Production - PS_Q")
            fig = px.density_heatmap(
                data_frame=df, y="DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK", x="BIOGAS_PRODUCTION_Q_DAY"
            )
            st.write(fig)
        
        with fig_col2:
            st.markdown("## Biogas Production - Date")
            fig2 = px.line(data_frame=df, y="BIOGAS_PRODUCTION_Q_DAY", x="Date")
            st.write(fig2)

        st.markdown("### Detailed Data View")
        st.dataframe(df)

    # User input for all columns
    new_data = {}
    new_data["Date"] = st.text_input("Enter Date")
    new_data["PS_Q_DAY"] = st.number_input("Enter PS_Q_DAY value")
    new_data["TPS_Q1_DAY"] = st.number_input("Enter TPS_Q1_DAY value")
    new_data["TWAS_DAF_QIN_DAY"] = st.number_input("Enter TWAS_DAF_QIN_DAY value")
    new_data["DIGESTED_SLUDGE_QOUT_DAY"] = st.number_input("Enter DIGESTED_SLUDGE_QOUT_DAY value")
    new_data["BIOGAS_PRODUCTION_Q_DAY"] = st.number_input("Enter BIOGAS_PRODUCTION_Q_DAY value")
    new_data["DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK"] = st.number_input("Enter DIG_SLUDGE_DEWATER_DS_AFTER_DEWATER_3_PER_WEEK value")

    if st.button("Add to CSV"):
        append_data(df, new_data)
        st.success("Data added successfully!")
