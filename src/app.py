import streamlit as st

from chat_engine import ChatEngine
from vector_index import VectorIndex


class App:
    def __init__(self, title="Local Chatbot"):
        self.app_title = title
        self.chat_engine = ChatEngine()
        self.vector_index = VectorIndex(
            db_path="./db",
            doc_path="./docs",
            collection_name="local-rag",
            model_name="BAAI/bge-base-en-v1.5",
            remove_old=True,
        )

        st.session_state.response = ""
        st.session_state.messages = []

    def run(self):
        st.title(self.app_title)

        self.draw_message_history()
        self.chat_input()

    def draw_message_history(self):
        if "message" in st.session_state.messages:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

    def chat_input(self):
        if prompt := st.chat_input("Say something"):
            with st.chat_message(name="user"):
                st.markdown(prompt)

            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.chat_message(name="assistant"):
                st.session_state.response_content = st.write_stream(
                    self.chat_engine.generate_response(prompt, self.vector_index)
                )

            st.session_state.messages.append(
                {"role": "assistant", "content": st.session_state.response_content}
            )


app = App()

app.run()
