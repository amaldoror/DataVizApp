import streamlit as st


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