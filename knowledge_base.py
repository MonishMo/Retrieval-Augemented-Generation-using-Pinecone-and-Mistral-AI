import time
import os

import logging

from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import pinecone

from pinecone import Pinecone, ServerlessSpec, PodSpec

logger = logging.getLogger(__name__)

class KnowledgeBase():

    def __init__(self):
        pinecone_api_key = os.environ['PINECONE_API_KEY']
        self.pc = Pinecone(api_key=pinecone_api_key)

    def create_index(self, index_name, use_serverless=None):
        try:
            if use_serverless:
                spec = ServerlessSpec(cloud='aws', region='us-west-2')
            else:
                spec = PodSpec(
                    environment="us-west4-gcp",
                    pod_type="p1.x1",
                    pods=1
                )
            if index_name not in self.pc.list_indexes().names():
                self.pc.create_index(
                    index_name,
                    dimension=384,
                    metric='cosine',
                    spec=spec
                )

                while not self.pc.describe_index(index_name).status['ready']:
                    time.sleep(1)

                logger.info("Index is created successfully")
            else:
                logger.info({"Index is already created"})

        except Exception as error:
            logger.error(f"Error while creating index: {index_name} Error: {error}")

    def delete_index(self, index_name):
        try:
            if index_name in self.pc.list_indexes().names():
                self.pc.delete_index(index_name)
            logger.info("Index is deleted successfully")
        except Exception as error:
            logger.error(f"Error while deleting index: {index_name} Error: {error}")

    def list_indexes(self, index_name=None):
        if index_name is not None and index_name in self.pc.list_indexes:
            return self.pc.describe_index(index_name)
        else:
            return self.pc.list_indexes()

    def upsert_from_documents(self, data, embeddings, index_name):
        try:
            if index_name not in self.pc.list_indexes().names():
                self.create_index(index_name)

            index = self.pc.Index(index_name)

            if not index.describe_index_stats()['total_vector_count']:
                PineconeVectorStore.from_documents(data, embeddings, index_name=index_name)
                logger.info(f"Data successfully added to the index {index_name}")
            else:
                logger.info("Index already have data. Use add text to append")
                return False

            return True

        except Exception as error:
            print(f"Error while inserting the data into Vector DB index {index_name}:  {error}")


    def get_from_existing_index(self, index_name, embeddings):
        docsearch = PineconeVectorStore.from_existing_index(index_name, embeddings)

        retriever = docsearch.as_retriever()

        return retriever