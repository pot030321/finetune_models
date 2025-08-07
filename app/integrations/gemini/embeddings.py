from typing import List
from langchain_core.embeddings import Embeddings
import google.generativeai as genai
import os

class GeminiEmbeddings(Embeddings):
    """Langchain-compatible Gemini embeddings that works with sync operations only"""
    
    def __init__(self, model: str = "models/text-embedding-004"):
        self.model = model
        
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []
        for text in texts:
            try:
                result = genai.embed_content(
                    model=self.model,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(result['embedding'])
            except Exception as e:
                print(f"Error creating embedding for text: {str(e)}")
                raise
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_query"
            )
            return result['embedding']
        except Exception as e:
            print(f"Error creating embedding for query: {str(e)}")
            raise
