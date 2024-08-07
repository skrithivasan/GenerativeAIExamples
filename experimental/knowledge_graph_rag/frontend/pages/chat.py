import streamlit as st
import requests
import json
import time

st.set_page_config(layout="wide")

st.title("Chat with your Knowledge Graph!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

response = requests.get("http://localhost:8000/chat/get-models/")
if response.status_code == 200:
    available_models = response.json()["models"]
else:
    st.error("Error fetching models.")
    available_models = []

with st.sidebar:
    llm = st.selectbox("Choose an LLM", available_models, index=available_models.index("mistralai/mixtral-8x7b-instruct-v0.1") if "mistralai/mixtral-8x7b-instruct-v0.1" in available_models else 0)
    st.write("You selected: ", llm)

with st.sidebar:
    use_kg = st.checkbox("Use knowledge graph")

user_input = st.chat_input("Can you tell me how research helps users to solve problems?")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        response = requests.post("http://localhost:8000/chat/chat/", json={"user_input": user_input, "use_kg": use_kg, "model_id": llm})
        if response.status_code == 200:
            response_data = response.json()
            context = response_data.get("context", "")
            assistant_response = response_data.get("assistant_response", "")
            for chunk in assistant_response.split(' '):
                full_response += chunk + ' '
                message_placeholder.markdown(full_response)
                time.sleep(0.05)  #
        else:
            st.error("Error processing the chat request.")

    st.session_state.messages.append({"role": "assistant", "content": full_response})
