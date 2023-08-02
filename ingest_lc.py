
"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import platform

from dotenv import load_dotenv
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

load_dotenv()


import os

def load_html_docs():
    """Load documents from web pages and return them."""
    if platform.system() == "Windows":
        loader = ReadTheDocsLoader("api.python.langchain.com/en/latest/", "utf-8-sig")
        print("\nusing utf-8-sig windows")
        print(f"Current working directory: {os.getcwd()}")
    else:
        loader = ReadTheDocsLoader("api.python.langchain.com/en/latest/", "utf-8-sig")
        print("\nusing utf-8-sig")
        print(f"Current working directory: {os.getcwd()}")

    raw_documents = loader.load()
    return raw_documents

def create_vectors_and_save(raw_documents):
    print("Raw documents length:", len(raw_documents)) # Print raw_documents length

    """Create vectors from raw documents and save them to a pickle file."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    documents = text_splitter.split_documents(raw_documents)
    print("Documents length:", len(documents)) # Print documents length
    print("Documents split into chunks.")

    embeddings = OpenAIEmbeddings()
    print("Embeddings:", embeddings) # Print embeddings if it's feasible
    print("OpenAI Embeddings created.")

    vectorstore = FAISS.from_documents(documents, embeddings)
    print("Vectorstore created from documents.")

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)
    print("Vectorstore saved to pickle file.")

def load_local_html_docs():
    """Load locally saved HTML documents and return them."""
    path = "api.python.langchain.com/en/latest/" # Adjust the path as needed
    if platform.system() == "Windows":
        loader = ReadTheDocsLoader(path, "utf-8-sig")
        print("\nusing utf-8-sig windows")
    else:
        loader = ReadTheDocsLoader(path, "utf-8-sig")
        print("\nusing utf-8-sig")

    raw_documents = loader.load()
    return raw_documents

if __name__ == "__main__":
    file = open("../../openaiapikey.txt", "r")
    openaikey = file.read()
    file.close()
    os.environ["OPENAI_API_KEY"]=openaikey
    raw_documents = load_local_html_docs()
    create_vectors_and_save(raw_documents)


# if __name__ == "__main__":
#     raw_documents = load_html_docs()
#     create_vectors_and_save(raw_documents)

