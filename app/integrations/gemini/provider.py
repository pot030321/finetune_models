from typing import List, Dict, Any
from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv('.env')

class GeminiProvider:
    def __init__(
        self,
        model: str = 'gemini',
        chat_model: str = 'gemini-1.5-flash',
        embedding_model: str = 'models/text-embedding-004'
    ):
        self.model = model
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.client = genai
        
    def embeddings_create(self, input: List[str]) -> Dict[str, Any]:
        """
        Create embeddings using Gemini API
        Returns a structure similar to OpenAI's response format
        """
        embeddings_data = []
        
        for i, text in enumerate(input):
            try:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                
                # Create a structure similar to OpenAI's response
                embedding_obj = type('Embedding', (), {
                    'embedding': result['embedding'],
                    'index': i,
                    'object': 'embedding'
                })()
                
                embeddings_data.append(embedding_obj)
                
            except Exception as e:
                print(f"Error creating embedding for text {i}: {str(e)}")
                raise
        
        # Return structure similar to OpenAI's response
        return type('EmbeddingResponse', (), {
            'data': embeddings_data,
            'model': self.embedding_model,
            'object': 'list',
            'usage': {
                'prompt_tokens': sum(len(text.split()) for text in input),
                'total_tokens': sum(len(text.split()) for text in input)
            }
        })()
    
    def get_chat_model(self):
        """Get chat model instance"""
        return genai.GenerativeModel(self.chat_model)

class SimpleGeminiEmbeddings:
    """Simple Gemini embeddings without Langchain dependencies"""
    
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

class GeminiProvider:
    def __init__(
        self,
        model: str = 'gemini',
        chat_model: str = 'gemini-1.5-flash',
        embedding_model: str = 'models/text-embedding-004'
    ):
        self.model = model
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        
        # Configure Gemini API
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.client = genai
        
    def embeddings_create(self, input: List[str]) -> Dict[str, Any]:
        """
        Create embeddings using Gemini API
        Returns a structure similar to OpenAI's response format
        """
        embeddings_data = []
        
        for i, text in enumerate(input):
            try:
                result = genai.embed_content(
                    model=self.embedding_model,
                    content=text,
                    task_type="retrieval_document"
                )
                
                # Create a structure similar to OpenAI's response
                embedding_obj = type('Embedding', (), {
                    'embedding': result['embedding'],
                    'index': i,
                    'object': 'embedding'
                })()
                
                embeddings_data.append(embedding_obj)
                
            except Exception as e:
                print(f"Error creating embedding for text {i}: {str(e)}")
                raise
        
        # Return structure similar to OpenAI's response
        return type('EmbeddingResponse', (), {
            'data': embeddings_data,
            'model': self.embedding_model,
            'object': 'list',
            'usage': {
                'prompt_tokens': sum(len(text.split()) for text in input),
                'total_tokens': sum(len(text.split()) for text in input)
            }
        })()
    
    def get_chat_model(self):
        """Get chat model instance"""
        return genai.GenerativeModel(self.chat_model)
