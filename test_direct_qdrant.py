"""
Test Qdrant connection directly
"""
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance

load_dotenv()

def test_direct_connection():
    url = os.environ["QDRANT_URL"]
    api_key = os.environ["QDRANT_API_KEY"]
    
    print(f"Testing connection to: {url}")
    print(f"Using API key: {api_key[:20]}...")
    
    try:
        # Test different connection configurations
        configs = [
            {"url": url, "api_key": api_key, "verify": False, "prefer_grpc": False},
            {"url": url, "api_key": api_key, "verify": False},
            {"url": url, "api_key": api_key}
        ]
        
        for i, config in enumerate(configs):
            print(f"\nTrying config {i+1}: {config}")
            try:
                client = QdrantClient(**config)
                
                # Test basic operations
                collections = client.get_collections()
                print(f"✅ Success! Found {len(collections.collections)} collections")
                
                # Try to create a test collection
                collection_name = "test_collection"
                try:
                    client.delete_collection(collection_name)
                    print(f"Deleted existing test collection")
                except:
                    pass
                
                client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                print(f"✅ Created test collection successfully")
                
                # Verify collection
                info = client.get_collection(collection_name)
                print(f"✅ Collection dimension: {info.config.params.vectors.size}")
                
                # Clean up
                client.delete_collection(collection_name)
                print(f"✅ Cleaned up test collection")
                
                return client
                
            except Exception as e:
                print(f"❌ Config {i+1} failed: {e}")
                continue
                
        print("❌ All configurations failed")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    client = test_direct_connection()
    if client:
        print("\n✅ Qdrant connection working!")
    else:
        print("\n❌ Could not connect to Qdrant")
