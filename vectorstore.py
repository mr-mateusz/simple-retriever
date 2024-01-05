from typing import Sequence

from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance, PointStruct
from typing_extensions import Self

from data_model import DocumentChunk, RetrievedDocumentChunk


class QdrantVectorstore:
    """Wrapper over Qdrant client."""

    def __init__(self, client: QdrantClient, collection_name: str) -> None:
        self.client = client
        self.collection_name = collection_name

    @classmethod
    def in_memory(cls, vector_size: int, collection_name: str = 'vectorstore') -> Self:
        """Create and initialise instance with qdrant in memory vectorstore."""
        client = QdrantClient(':memory:')

        client.create_collection(collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.DOT))

        return cls(client, collection_name)

    def add(self, documents: Sequence[DocumentChunk]) -> None:
        points = [PointStruct(id=d.id, vector=d.vector, payload={'text': d.text, 'metadata': d.metadata})
                  for d in documents]

        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )

    def search(self, embedding: Sequence[float], limit: int = 3) -> list[RetrievedDocumentChunk]:
        result = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=limit
        )

        return [RetrievedDocumentChunk(text=r.payload['text'], metadata=r.payload['metadata'], id=r.id, score=r.score)
                for r in result]

    @property
    def total_chunks(self):
        return self.client.count(self.collection_name)
