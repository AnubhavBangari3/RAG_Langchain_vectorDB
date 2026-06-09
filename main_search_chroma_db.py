from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

results = vector_store.similarity_search(
    "What is a vector database?",
    k=2
)

for doc in results:
    print(doc.page_content)