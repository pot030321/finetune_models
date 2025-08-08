"""
Script to reset Qdrant collection with correct dimensions for Gemini embeddings
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

load_dotenv()

def reset_collection():
    # Initialize Qdrant client directly
    client = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"]
    )
    
    collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'documents')
    
    try:
        # Try to delete existing collection
        print(f"Attempting to delete collection: {collection_name}")
        client.delete_collection(collection_name)
        print(f"Successfully deleted collection: {collection_name}")
    except Exception as e:
        print(f"Collection {collection_name} does not exist or error deleting: {e}")
    
    try:
        # Create new collection with correct dimension for Gemini (768)
        print(f"Creating new collection: {collection_name} with dimension 768")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        print(f"Successfully created collection: {collection_name}")
        
        # Verify collection
        collection_info = client.get_collection(collection_name)
        print(f"Collection created with vector size: {collection_info.config.params.vectors.size}")
        
    except Exception as e:
        print(f"Error creating collection: {e}")

if __name__ == "__main__":
    reset_collection()
