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

linkedin = "https://raw.githubusercontent.com/amaldoror/DataVizApp/main/img/linkedin.gif"
email = "https://raw.githubusercontent.com/amaldoror/DataVizApp/main/img/email.gif"
newsletter = (
    "https://raw.githubusercontent.com/amaldoror/DataVizApp/main/img/letter.gif"
)

uptime = "https://uptime.betterstack.com/status-badges/v1/monitor/196o6.svg"

st.sidebar.caption(
    f"""
        <div style='display: flex; align-items: center;'>
            <a href = 'https://www.linkedin.com/'>
            <img src='{linkedin}' style='width: 35px; height: 35px; margin-right: 25px;'></a>
            <a href = 'mailto:sahir@adrian.morgenthal92@gmail.com'>
            <img src='{email}' style='width: 28px; height: 28px; margin-right: 25px;'></a>
            <a href = 'https://www.linkedin.com/'>
            <img src='{newsletter}' style='width: 28px; height: 28px; margin-right: 25px;'></a>

        </div>
        <br>
        <a href = 'https://exifa.betteruptime.com/'><img src='{uptime}'></a>

        """,
    unsafe_allow_html=True,
)

pg.run()
