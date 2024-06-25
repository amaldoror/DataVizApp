import streamlit as st
import streamlit.components.v1 as components
from components.particles import particles_js


if "show_animation" not in st.session_state:
    st.session_state.show_animation = True
if "messages" not in st.session_state:
    st.session_state.messages = []


# Toggle Chatbox
# show_chat = st.checkbox("Show Chatbox")


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Write your message")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.write(f"User: {prompt}")

if st.session_state.show_animation:
    components.html(particles_js, height=370, scrolling=False)
