import ollama
import streamlit as st

from vector_index import VectorIndex


class ChatEngine:
    def __init__(self, chat_model: str):
        self.chat_model = chat_model

    def generate_response(self, prompt: str, vector_index: VectorIndex):
        docs = vector_index.collection.query(
            query_texts=[prompt],
            n_results=5,
        )
        docs_str = "\n\n".join(docs)

        system_message = [
            {
                "role": "system",
                "content": f"""
                    You are a locally running LLM based chatbot.
                    You have to give answer based on the following source documents:
                    <documents>{docs_str}</documents>
                """,
            }
        ]

        messages = system_message + st.session_state.messages

        response = ollama.chat(model=self.chat_model, messages=messages, stream=True)

        for chunk in response:
            yield chunk["message"]["content"]
