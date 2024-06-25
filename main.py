import pandas as pd
import pypdf
import streamlit as st
from streamlit_multipage import MultiPage


st.set_page_config(page_title="DataVizApp", layout="wide")


home = st.Page("pages/home.py", title="Home", icon=":material/home:")
page1 = st.Page("pages/page1.py", title="Chat", icon=":material/chat:")
page2 = st.Page("pages/page2.py", title="LaTeX", icon=":material/functions:")
page3 = st.Page("pages/page3.py", title="About", icon=":material/info:")

pg = st.navigation([home, page1, page2, page3])

st.sidebar.header("Info")
st.sidebar.write("Lorem ipsum dolor sit amet")

# Render current page content
pg.run()