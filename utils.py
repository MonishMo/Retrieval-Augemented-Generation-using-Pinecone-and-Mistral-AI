import sys
import csv
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA

from langchain_community.document_loaders import DirectoryLoader

from langchain_community.document_loaders import UnstructuredHTMLLoader

class Processor():
    def __init__(self):
        # csv.field_size_limit(sys.maxsize)
        self.pinecone_api_key = os.environ["PINECONE_API_KEY"]
        self.embedding_model = os.environ["EMBEDDING_MODEL"]

    def load_document(self, document_filepath):
        loader = UnstructuredHTMLLoader(document_filepath)
        data = loader.load()

        return data

    def document_splitter(self, documents):
        chunk_size = os.environ['chunk_size']
        chunk_overlap = os.environ['chunk_overlap']
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        docs = text_splitter.split_documents(documents)

        return docs

    def get_embeddings(self):
        modelPath = self.embedding_model
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}

        embeddings = HuggingFaceEmbeddings(
            model_name=modelPath,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )

        return embeddings



