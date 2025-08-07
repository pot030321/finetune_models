import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from typing import Any, Dict, List
from qdrant_client.http.models import VectorParams, Distance, models
from qdrant_client.conversions.common_types import VectorParams

load_dotenv(".env")


class Qdrant:
    def __init__(
        self,
        url: str = os.environ["QDRANT_URL"],
        api_key: str = os.environ["QDRANT_API_KEY"],
    ):
        self.client = QdrantClient(
            url=url, 
            api_key=api_key, 
            verify=False,
            prefer_grpc=False
        )

    def create_collection(
        self, collection_name: str, size: int = 768, distance: str = Distance.COSINE
    ):
        try:
            self.client.get_collection(collection_name)
        except Exception as e:
            if "not found" in str(e).lower():
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=size, distance=distance, on_disk=True
                    ),
                    quantization_config=models.ScalarQuantization(
                        scalar=models.ScalarQuantizationConfig(
                            type=models.ScalarType.INT8, always_ram=True
                        )
                    ),
                )
                self.create_index(
                    collection_name=collection_name,
                    field_name="key",
                    field_schema="keyword",
                )
            else:
                raise e

        return True

    def create_index(self, collection_name: str, field_name: str, field_schema: str):
        return self.client.create_payload_index(
            collection_name,
            field_name=field_name,
            field_schema=field_schema,
        )

    def add_points(
        self, collection_name: str, points: Dict[str, Any], wait: bool = True
    ):
        return self.client.upsert(collection_name, points, wait)

    def remove_points_by_keys(self, collection_name: str, keys: List[str]):
        return self.client.delete(
            collection_name,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    should=[
                        models.FieldCondition(
                            key="key",
                            match=models.MatchValue(value=key),
                        )
                        for key in keys
                    ],
                )
            ),
        )
