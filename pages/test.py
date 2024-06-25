import streamlit as st
import pandas as pd
import numpy as np
import time

# CSS for gradient background
gradient_css = """
<style>
    .stApp {
        background: linear-gradient(45deg, #0207CB, #CB7C02);
        background-attachment: fixed;
    }
</style>
"""

# Inject CSS
st.markdown(gradient_css, unsafe_allow_html=True)


# Slider
x = st.slider("Select x value", 1, 10, 1)
y = st.slider("Select y value", 1, 10, 1)


df = pd.DataFrame(np.random.randn(15, 3), columns=(["A", "B", "C"]))

chart_type = st.selectbox(
    'Choose a chart type',
    ['Line Chart', 'Bar Chart', 'Area Chart'],
    index=0
)
generate = st.button("Generate", use_container_width=1)
if generate:
    if chart_type == 'Line Chart':
        my_data_element = st.line_chart(df)
    elif chart_type == 'Bar Chart':
        my_data_element = st.bar_chart(df)
    elif chart_type == 'Area Chart':
        my_data_element = st.area_chart(df)
    for tick in range(x):
        time.sleep(.5)
        add_df = pd.DataFrame(np.random.randn(1, 3), columns=(["A", "B", "C"]))
        my_data_element.add_rows(add_df)


