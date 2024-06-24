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
header = {
    st.title(':blue[amaldoror Data Visualization App]'),
    st.header(':blue[Header]'),
    st.subheader(':blue[Subheader]'),
    st.divider()
}


# ---------------------- File Uploader ----------------------
def read_file(file):
    file_type = file.type
    file_name = file.name

    # Debugging information
    st.write(f"Detected file type: {file_type}")
    st.write(f"File name: {file_name}")

    try:
        # Check if file is CSV based on MIME type or file extension
        if file_type == 'text/csv' or file_name.endswith('.csv'):
            return pd.read_csv(file, encoding='utf-8')
        elif file_type == 'application/vnd.ms-excel' or file_name.endswith('.xls'):
            return pd.read_excel(file, engine='xlrd')
        elif (file_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
              or file_name.endswith('.xlsx')):
            return pd.read_excel(file, engine='openpyxl')
        elif file_type == 'application/json' or file_name.endswith('.json'):
            return pd.read_json(file, encoding='utf-8')
        elif file_type == 'application/pdf' or file_name.endswith('.pdf'):
            reader = pypdf.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text.encode('utf-8', errors='ignore').decode('utf-8')
        else:
            return None
    except UnicodeDecodeError:
        st.error("There was an error decoding the file. Please ensure the file is in UTF-8 format.")
        return None


# File uploader
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx', 'xls', 'json', 'pdf'])


def show_basic_stats(data):
    # Show basic statistics
    st.write("Basic Statistics")
    st.write(data.describe())


def select_features(columns):
    st.write("Select features for plotting")
    features = st.multiselect('Choose features for plotting', columns)
    return features


def plot_features(features):
    st.write("Data Visualization")
    chart_type = st.selectbox(
        'Choose a chart type',
    )
    if chart_type == 'Line Chart':
        st.line_chart(df)
    elif chart_type == 'Bar Chart':
        st.bar_chart(df)
    elif chart_type == 'Area Chart':
        st.area_chart(df)
    elif chart_type == 'Histogram':
        # TODO: Implement Histogram
        st.error('Histogram not supported yet')
    elif chart_type == 'Scatterplot':
        # TODO: Implement Scatterplot
        st.error('Scatterplot not supported yet')


def select_columns():
    # Select columns for plotting
    st.write("Select columns for plotting")
    columns = df.columns.tolist()
    x_column = st.selectbox('Choose a column for the x-axis:', columns)
    y_column = st.selectbox('Choose a column for the y-axis:', columns)
    return x_column, y_column


def plot_data(df, x_column, y_column):
    st.write("Data Visualization")
    chart_type = st.selectbox('Choose a chart type', ['Line Chart', 'Bar Chart', 'Area Chart'])
    if chart_type == 'Line Chart':
        st.line_chart(df[[x_column, y_column]].set_index(x_column))
    elif chart_type == 'Bar Chart':
        st.bar_chart(df[[x_column, y_column]].set_index(x_column))
    elif chart_type == 'Area Chart':
        st.area_chart(df[[x_column, y_column]].set_index(x_column))


if uploaded_file is not None:
    df = read_file(uploaded_file)
    if isinstance(df, pd.DataFrame):
        st.write(df)
        show_basic_stats(df)
        x_column, y_column = select_columns()
        plot_data(df, x_column, y_column)
    elif isinstance(df, str):
        st.write(df)
    else:
        st.error("Unsupported file format")


def select_model():
    st.write("Select model for plotting")
    model_choice = st.selectbox(
        'Choose a model',
        ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'Neural Net']
    )
    return model_choice


def train_model(model_choice, df):
    x = df.drop(columns=df.columns[-1])
    y = df[df.columns[-1]]
    x_train, x_test, y_train, y_test = preprocess_data(x, y)
    match model_choice:
        case 'Logistic Regression':
            model = train_lr(x_train, x_test, y_train, y_test)
        case 'Decision Tree':
            model = train_dt(x_train, x_test, y_train, y_test)
        case 'Random Forest':
            model = train_rf(x_train, x_test, y_train, y_test)
        case 'SVM':
            model = train_svm(x_train, x_test, y_train, y_test)
        case 'Neural Net':
            model = train_snn(x_train, x_test, y_train, y_test)
        case _:
            st.error("Model not supported")
            return None
    print_model_info(model, x_test, y_test)
    return model


model_choice = select_model()
if uploaded_file is not None and isinstance(df, pd.DataFrame):
    model = train_model(model_choice, df)


# Slider
x = st.slider("Select a value")
st.write(f"Selected value: {x}")

# LaTeX Button
if st.button('LaTeX'):
    st.latex(r'''
        a + ax + a x^2 + a x^3 + \cdots + a x^{n-1} =
        \sum_{k=0}^{n-1} ax^k =
        a \left(\frac{1-x^{n}}{1-x}\right)
        ''')

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Toggle Chatbox
show_chat = st.checkbox("Show Chatbox")

if show_chat:
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Write your message")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.write(f"User: {prompt}")