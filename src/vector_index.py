import torch
import chromadb
import streamlit as st

from chromadb import Documents, EmbeddingFunction, Embeddings, Collection

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name, device):
        super().__init__(self)
        self.model = get_embedding_model(model_name, device)
        
    def __call__(self, input: Documents) -> Embeddings:
        return [self.model.get_text_embedding(d) for d in input]

@st.cache_resource
def get_embedding_model(model, device):
    return HuggingFaceEmbedding(model_name=model, device=device)

class VectorIndex:
    def __init__(
        self,
        db_path: str,
        doc_path: str,
        collection_name: str,
        model_name: str,
        remove_old=False,
    ):
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        self.client = chromadb.PersistentClient(db_path)

        if remove_old and collection_name in [
            c.name for c in self.client.list_collections()
        ]:
            self.client.delete_collection(collection_name)

        self.collection = self.client.get_or_create_collection(
            collection_name,
            embedding_function=MyEmbeddingFunction(model_name, self.device),
            metadata={"hnsw:space": "cosine"},
        )

        self.populate_collection(doc_path, self.collection)

    def populate_collection(self, doc_path: str, collection: Collection):
        nodes = get_documents(doc_path)

        docs = [n.text for n in nodes]
        ids = [f"{i}id" for i in range(len(docs))]

        self.collection.add(documents=docs, ids=ids)

@st.cache_data
def get_documents(path: str):
    docs = SimpleDirectoryReader(path).load_data()
    text_splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=32)

    for doc in docs:
        doc.doc_id = doc.metadata["file_name"].split(".")[0]

    return text_splitter.get_nodes_from_documents(docs)


