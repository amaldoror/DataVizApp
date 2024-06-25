import streamlit as st


def home():
    st.title("Chat")


# ------------------------- Header -------------------------
header = {
    st.title(':blue[amaldoror Data Visualization App]'),
    st.header(':blue[Header]'),
    st.subheader(':blue[Subheader]'),
    st.divider()
}

# Slider
x = st.slider("Select a value")
st.write(f"Selected value: {x}")
