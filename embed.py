from langchain_ollama import OllamaEmbeddings

def embed_it():
    embeddings = OllamaEmbeddings(model='nomic-embed-text')
    return embeddings