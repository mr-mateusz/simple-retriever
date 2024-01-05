from typing_extensions import Self

from article_parser import parse_article_base, split_title
from data_model import DocumentChunk, RetrievedDocumentChunk
from embedding_encoder import Encoder
from text_splitter import to_chunks
from vectorstore import QdrantVectorstore


class QAService:
    """Class which encapsulates whole retrieval system logic."""

    def __init__(self, encoder: Encoder, vectorstore: QdrantVectorstore) -> None:
        self.encoder = encoder
        self.vectorstore = vectorstore

    @classmethod
    def from_default(cls) -> Self:
        encoder = Encoder()
        vectorstore = QdrantVectorstore.in_memory(encoder.vector_size)
        return cls(encoder, vectorstore)

    def add_html_article(self, text: str) -> None:
        """Parse the article, split into chunks, create embedding vectors and index into vectorstore."""
        text = parse_article_base(text)
        title, text = split_title(text)
        texts = to_chunks(text)

        embeddings = self.encoder.embed_documents(texts, [title] * len(texts))

        doc_chunks = [DocumentChunk(text=t, vector=e, metadata={'title': title}) for t, e in zip(texts, embeddings)]

        self.vectorstore.add(doc_chunks)

    def query(self, query: str, limit: int = 3) -> list[RetrievedDocumentChunk]:
        """Find parts of documents relevant to the query."""
        query_embedding = self.encoder.embed_query(query)

        results = self.vectorstore.search(query_embedding, limit)

        return results
