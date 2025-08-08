"""
Check available Gemini models
"""
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

def check_available_models():
    # Configure Gemini API
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("GEMINI_API_KEY not found in environment variables")
        return
    
    genai.configure(api_key=api_key)
    
    try:
        models = genai.list_models()
        
        print("Available Gemini models:")
        print("=" * 50)
        
        chat_models = []
        embedding_models = []
        
        for model in models:
            print(f"Model: {model.name}")
            print(f"  Display Name: {model.display_name}")
            print(f"  Description: {model.description}")
            print(f"  Supported methods: {[method.name for method in model.supported_generation_methods]}")
            print()
            
            if 'generateContent' in [method.name for method in model.supported_generation_methods]:
                chat_models.append(model.name)
            
            if 'embedContent' in [method.name for method in model.supported_generation_methods]:
                embedding_models.append(model.name)
        
        print("\nChat Models (supports generateContent):")
        for model in chat_models:
            print(f"  - {model}")
            
        print("\nEmbedding Models (supports embedContent):")
        for model in embedding_models:
            print(f"  - {model}")
            
    except Exception as e:
        print(f"Error checking models: {e}")

if __name__ == "__main__":
    check_available_models()
