"""Load html from files, clean up, split, ingest into Weaviate."""
import pickle
import os
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS


def ingest_docs():
    """Get documents from web pages."""
#    loader = ReadTheDocsLoader("langchain.readthedocs.io/en/latest/")
    loader = PyPDFDirectoryLoader("example_data/")
    raw_documents = loader.load_and_split()

#    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=50,
    )
    documents = text_splitter.split_documents(raw_documents)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save vectorstore
    with open("vectorstore.pkl", "wb") as f:
        pickle.dump(vectorstore, f)


if __name__ == "__main__":
    file = open("../../openaiapikey.txt", "r")
    openaikey = file.read()
    file.close()
    os.environ["OPENAI_API_KEY"]=openaikey
    ingest_docs()
