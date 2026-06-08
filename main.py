from dotenv import load_dotenv
from importlib.metadata import version

from langchain_ollama import ChatOllama

load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")


print(f"langchain-core version: {core_version}")
print(f"langgraph version: {lg_version}")


def main():
    # Test Ollama local LLM
    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0,
    )

    response = llm.invoke("Say 'setup complete!' in one word")

    print(f"Response from Ollama: {response.content}")
    print("Setup complete!")


if __name__ == "__main__":
    main()