import streamlit as st
import streamlit.components.v1 as components
from components.particles import particles_js


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

if "show_animation" not in st.session_state:
    st.session_state.show_animation = True
if "messages" not in st.session_state:
    st.session_state.messages = []


# Toggle Chatbox
# show_chat = st.checkbox("Show Chatbox")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input field for new messages
prompt = st.chat_input("Write your message")
if prompt:
    # Add the new message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Respond to the user message
    response = f"Echo: {prompt}"
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)

if st.session_state.show_animation:
    components.html(particles_js, height=370, scrolling=False)
