import streamlit as st
import pandas as pd
import numpy as np
import pypdf
import json
from MachineLearning import (preprocess_data,
                             plot_sample_images,
                             train_lr,
                             train_dt,
                             train_rf,
                             train_svm,
                             train_snn,
                             print_model_info)

# ------------------------- Header -------------------------

st.title(':blue[amaldoror Data Visualization App]')
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
st.write("Line Chart:")
st.line_chart(df)

# Slider
st.write("Slider:")
x = st.slider("Select a value")
st.write(f"Selected value: {x}")

# Selectbox
st.write("Selectbox:")
option = st.selectbox(
    'Select:',
    df.columns
)
st.write(f'You selected: {option}')

# Button
st.write("Button:")
if st.button('Button'):
    st.write('Hello, Streamlit!')


# ---------------------- File Uploader ----------------------

def read_file(file):
    file_type = file.type
    match file_type:
        case 'text/csv':
            return pd.read_csv(file)
        case 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            return pd.read_excel(file)
        case 'application/json':
            return pd.read_json(file)
        case 'application/pdf':
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        case default:
            return None


# File uploader
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'json', 'pdf'])
if uploaded_file is not None:
    data = read_file(uploaded_file)
    if isinstance(data, pd.DataFrame):
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
    elif isinstance(data, str):
        st.write(data)
    else:
        st.error("Unsupported file format")

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
