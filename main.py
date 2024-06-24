import streamlit as st
import pandas as pd
import numpy as np
from MachineLearning import print_model

# ------------------------- Header -------------------------

st.title(':blue[amaldoror Multipurpose App]')
st.header(':blue[Header]')
st.subheader(':blue[Subheader]')
st.divider()

# ------------------------- Data -------------------------

df = pd.DataFrame(
        {
           "x-Achse": np.random.randn(20),
           "y-Achse": np.random.randn(20),
           "Legende": np.random.choice(["A", "B", "C"], 20),
        }
    )
st.line_chart(df, x="x-Achse", y="y-Achse", color="Legende")

st.divider()

# Line chart
st.write("Here is a line chart:")
st.line_chart(df)

# Slider
st.write("Here is a slider:")
x = st.slider("Select a value")
st.write(f"Selected value: {x}")

# Selectbox
st.write("Here is a selectbox:")
option = st.selectbox(
    'Which number do you like best?',
    df.columns
)
st.write(f'You selected: {option}')

# Button
st.write("Here is a button:")
if st.button('Say hello'):
    st.write('Hello, Streamlit!')

# ---------------------- File Uploader ----------------------

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write(data)

    # Show basic statistics
    st.write("Basic Statistics")
    st.write(data.describe())

    # Plotting
    st.write("Data Visualization")
    chart_type = st.selectbox(
        'Choose a chart type',
        ['Line Chart', 'Bar Chart', 'Area Chart']
    )

    if chart_type == 'Line Chart':
        st.line_chart(data)
    elif chart_type == 'Bar Chart':
        st.bar_chart(data)
    elif chart_type == 'Area Chart':
        st.area_chart(data)

# ------------------------- Tasks -------------------------

st.markdown('''
- [x] Task 1
- [ ] Task 2
- [ ] Task 3
    ''')
st.divider()

# ------------------------- LaTeX -------------------------

st.latex(r'''
    a + ax + a x^2 + a x^3 + \cdots + a x^{n-1} =
    \sum_{k=0}^{n-1} ax^k =
    a \left(\frac{1-x^{n}}{1-x}\right)
    ''')

st.divider()

# ------------------------- Chat -------------------------

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Write your message")
if prompt:
    st.write(f"User: {prompt}")
