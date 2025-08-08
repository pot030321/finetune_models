"""
Test Qdrant connection and reset collection
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_and_reset():
    try:
        from app.core.database.qdrant import Qdrant
        
        print("Testing Qdrant connection...")
        qdrant = Qdrant()
        
        collection_name = os.getenv('QDRANT_COLLECTION_NAME', 'documents')
        
        # Check if collection exists
        try:
            collection_info = qdrant.client.get_collection(collection_name)
            print(f"Collection {collection_name} exists with dimension: {collection_info.config.params.vectors.size}")
            
            # Delete if wrong dimension
            if collection_info.config.params.vectors.size != 768:
                print(f"Wrong dimension, deleting collection...")
                qdrant.client.delete_collection(collection_name)
                print(f"Deleted collection {collection_name}")
            else:
                print("Collection has correct dimension (768)")
                return
                
        except Exception as e:
            print(f"Collection does not exist: {e}")
        
        # Create new collection
        print(f"Creating collection {collection_name} with dimension 768...")
        qdrant.create_collection(collection_name=collection_name, size=768)
        print(f"Successfully created collection {collection_name}")
        
        # Verify
        collection_info = qdrant.client.get_collection(collection_name)
        print(f"Verified: Collection {collection_name} has dimension {collection_info.config.params.vectors.size}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_and_reset()
