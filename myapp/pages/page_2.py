# Contents of ~/my_app/pages/page_2.py
import streamlit as st
import streamlit as st
import numpy as np
import pandas as pd

st.markdown("# Page 2 ❄️")
st.sidebar.markdown("# Page 2 ❄️")

# Include tabs in the page
tab1, tab2 = st.tabs(["📈 Chart", "🗃 Data"])
data = np.random.randn(10, 1)

with tab1:
    st.header("Nice Chart")
    st.subheader("A tab with a chart")
    st.line_chart(data)
    st.image("https://static.streamlit.io/examples/cat.jpg", width=200)
    

tab2.subheader("A tab with the data")
tab2.write(data)