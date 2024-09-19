# coda-local-rag-demo

Locally runnable RAG application demo.

## How To

### 1. Install Ollama

Install [ollama](https://ollama.com/) and run following command:

    ollama serve

### 2. Add Documents

The application is using `/docs` directory as source for document files. You can add some `*.pdf` files there that you're interested to use in your RAG application.

### 3. Run Streamlit Application

#### Option 1: Python

Run following commands:

    python3 -m venv venv
    source /venv/bin/activate
    pip install -requirements.txt
    streamlit run src/app.py

#### Option 2: Docker

Run followin commands:

    docker build -t local-rag .
    docker run -p 8501:8501 local-rag
