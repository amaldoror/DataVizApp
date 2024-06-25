import streamlit as st


st.set_page_config(page_title="DataVizApp", layout="centered", initial_sidebar_state="expanded")


home = st.Page("pages/home.py", title="Home", icon=":material/home:")
page1 = st.Page("pages/page1.py", title="Chat", icon=":material/chat:")
page2 = st.Page("pages/page2.py", title="LaTeX", icon=":material/functions:")
page3 = st.Page("pages/page3.py", title="About", icon=":material/info:")
test = st.Page("pages/test.py", title="Test", icon=":material/bug_report:")

pg = st.navigation([home, page1, page2, page3, test])

st.sidebar.header("Info")
st.sidebar.write("Lorem ipsum dolor sit amet")


pg.run()
