from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def main():

    # Sample documents
    documents = [

        Document(
            page_content="LangChain is a framework for building LLM applications.",
            metadata={"source": "langchain"}
        ),

        Document(
            page_content="ChromaDB is a vector database used in RAG systems.",
            metadata={"source": "chroma"}
        ),

        Document(
            page_content="Ollama allows running LLMs locally.",
            metadata={"source": "ollama"}
        ),
    ]

    # Local embedding model
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    # Create vector database
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    print("Documents stored successfully!")



if __name__ == "__main__":
    main()