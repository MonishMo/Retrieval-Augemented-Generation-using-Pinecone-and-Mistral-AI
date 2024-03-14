import os

from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
api_key = "I8PAB6hrXLBc0JGr4szaG2Gk9Y0F6SVQ"

class Model():
    def __init__(self):
        self.model_api_key = os.environ['MODEL_API_KEY']
        self.client = MistralClient(api_key=self.model_api_key)

    def inference(self, query, context):
        prompt = f"""Given the context\n {context}, I would like answer me by understanding the context
        Question: {query}

        """

        messages = [
            ChatMessage(role="user", content=prompt)
        ]

        chat_response = self.client.chat(
            model="mistral-medium",
            messages=messages,
            temperature=0.1
        )

        result = chat_response.choices[0].message.content

        return result