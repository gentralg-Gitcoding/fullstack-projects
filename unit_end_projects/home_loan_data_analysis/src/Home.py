import streamlit as st
import pandas as pd

st.title("Credit Card Customer Analysis")

st.markdown("""
This dashboard explores customer loan to debt ratios,
    and predicts if a user is prone to defaulting.
""")

#Cache expensive computations or potential ones
LOAD_URL = ("https://media.githubusercontent.com/media/gperdrizet/FSA_devops/refs/heads/main/data/unit4/Home_Loan.csv")
@st.cache_data
def load_data():
    return pd.read_csv(LOAD_URL)

df = load_data()

#Show summary overview of what the data is(Loading the CSV)
st.subheader("Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Customers", df.shape[0])
col2.metric("Features", df.shape[1])

#Show optional raw data(Sample Overview)
with st.expander("View raw data(top 100 rows)"):
    st.dataframe(df.head(100))