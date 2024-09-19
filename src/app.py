import streamlit as st

from chat_engine import ChatEngine
from vector_index import VectorIndex

@st.cache_resource
def get_chat_engine(chat_model: str):
    st.session_state.chat_engine = ChatEngine(chat_model)

@st.cache_resource
def get_vector_index(collection_name: str, model_name: str):
    st.session_state.vector_index = VectorIndex(
        db_path="./db",
        doc_path="./docs",
        collection_name=collection_name,
        model_name=model_name,
        remove_old=True,
    )

get_chat_engine("llama3.1:8b")
get_vector_index("local-rag", "BAAI/bge-base-en-v1.5")

if "response" not in st.session_state:
    st.session_state.response = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Local Chatbot")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Say something"):
    with st.chat_message(name="user"):
        st.markdown(prompt)

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message(name="assistant"):
        st.session_state.response_content = st.write_stream(
            st.session_state.chat_engine.generate_response(prompt, st.session_state.vector_index)
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": st.session_state.response_content}
    )