import logging
import os

from flask import Flask, request, jsonify

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI

from knowledge_base import KnowledgeBase
from utils import Processor
from models import Model



app = Flask(__name__)

logging.basicConfig(filename='chat_bot.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


@app.route('/chat', methods=['POST'])
def process_pdf():
    request_json = request.json

    if not request_json:
        return jsonify({"Status": 400, "Message": "Bad Request"}), 400
    try:
        index_name = os.environ['INDEX_NAME']
        llm_api_key = os.environ["LLM_API_KEY"]

        query = str(request_json['query'])

        kb = KnowledgeBase()
        processor = Processor()
        # model_obj = Model()

        embeddings = processor.get_embeddings()
        retriever = kb.get_from_existing_index(index_name, embeddings)

        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        # model = ChatOpenAI(openai_api_key=model_api_key, temperature=0.1)
        model = ChatMistralAI(mistral_api_key=llm_api_key)

        chain = (
                {"context": retriever, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
        )

        result = chain.invoke(query)

        return jsonify({"message": result}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/add_new_data', methods=['POST'])
def add_data(request):
    request_json = request.json

    if not request_json:
        return jsonify({"Status": 400, "Message": "Bad Request"}), 400

    try:
        index_name = request_json['Index']
        file_path = request_json['FilePath']
        kb = KnowledgeBase()
        processor = Processor()

        data = processor.load_document(file_path)
        documents = processor.document_splitter(data)
        embeddings = processor.get_embeddings()
        result = kb.upsert_from_documents(documents, embeddings, index_name)

        if result:
            return jsonify({"message": f"Data added successfully to the Vector DB. Index: {index_name}"}), 200

        return jsonify({"message": f"Error while added data to the vector DB"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)