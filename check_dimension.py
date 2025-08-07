"""
Check Gemini embedding dimension
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def check_gemini_dimension():
    # Configure Gemini API
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("GEMINI_API_KEY not found in environment variables")
        return
    
    genai.configure(api_key=api_key)
    
    try:
        # Test embedding
        result = genai.embed_content(
            model='models/text-embedding-004',
            content='test text for dimension check',
            task_type='retrieval_document'
        )
        
        dimension = len(result['embedding'])
        print(f"Gemini text-embedding-004 dimension: {dimension}")
        
        # Also check a different model if available
        models = genai.list_models()
        embedding_models = [m for m in models if 'embed' in m.name.lower()]
        print(f"Available embedding models: {[m.name for m in embedding_models]}")
        
    except Exception as e:
        print(f"Error checking dimension: {e}")

if __name__ == "__main__":
    check_gemini_dimension()
