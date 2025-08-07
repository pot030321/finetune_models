from typing import List
from dotenv import load_dotenv
import os

import openai
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

load_dotenv('.env')

openai_client = openai.Client(
    api_key=os.getenv('OPENAI_API_KEY'),
)

class GPTProvider:
    def __init__(
        self,
        model: str = 'gpt',
        chat_model: str = 'gpt-4o',
        embedding_model: str = 'text-embedding-3-small'
    ):
        self.model = model
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.embeddings = self._get_embeddings()
        self.chat_model_instance = self._get_chat_model()

    def _get_embeddings(self):
        return OpenAIEmbeddings(model=self.embedding_model)

    def _get_chat_model(self):
        return ChatOpenAI(model=self.chat_model, temperature=0, streaming=True)
    
    def embeddings_create(self, input: List[str]):
        return openai_client.embeddings.create(input=input, model=self.embedding_model)