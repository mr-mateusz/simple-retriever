from enum import Enum

from typing_extensions import Self

from article_parser import parse_article_base, split_title, parse_faq
from data_model import DocumentChunk, RetrievedDocumentChunk
from embedding_encoder import Encoder
from text_splitter import to_chunks
from vectorstore import QdrantVectorstore


class AnswerType(Enum):
    RELEVANT_CHUNKS = 'relevant_chunks'
    SIMILAR_QUESTION = 'similar_question'


class QAService:
    """Class which encapsulates whole retrieval system logic."""

    def __init__(self, encoder: Encoder, vectorstore: QdrantVectorstore, vectorstore_faq: QdrantVectorstore) -> None:
        self.encoder = encoder
        self.vectorstore = vectorstore
        self.vectorstore_faq = vectorstore_faq

    @classmethod
    def from_default(cls) -> Self:
        encoder = Encoder()
        vectorstore = QdrantVectorstore.in_memory(encoder.vector_size)

        faq_collection_name = 'vectorstore_faq'
        vectorstore_faq = QdrantVectorstore(vectorstore.client, faq_collection_name)
        vectorstore_faq.create_collection(encoder.vector_size)

        return cls(encoder, vectorstore, vectorstore_faq)

    def add_text(self, text: str, title: str) -> None:
        """Split text into chunks, create embedding vectors and index into vectorstore."""
        texts = to_chunks(text)
        embeddings = self.encoder.embed_documents(texts, [title] * len(texts))
        doc_chunks = [DocumentChunk(text=t, vector=e, metadata={'title': title}) for t, e in zip(texts, embeddings)]
        self.vectorstore.add(doc_chunks)

    def add_html_article(self, text: str) -> None:
        """Parse the article, split into chunks, create embedding vectors and index into vectorstore."""
        text = parse_article_base(text)
        title, text = split_title(text)
        return self.add_text(text, title)

    def add_html_faq(self, text: str, index_answers: bool = False) -> None:
        """Parse faq article, create embedding vectors and index into faq vectorstore."""
        qa_pairs = parse_faq(text)

        embeddings = [self.encoder.embed_query(pair.question) for pair in qa_pairs]

        chunks = [DocumentChunk(text=pair.question, vector=e,
                                metadata={'title': '', 'answer': pair.answer, 'category': pair.category})
                  for pair, e in zip(qa_pairs, embeddings)]

        self.vectorstore_faq.add(chunks)

        if not index_answers:
            return

        if index_answers:
            for pair in qa_pairs:
                self.add_text(pair.answer, pair.question)

    def query(self, query: str, limit: int = 3) -> list[RetrievedDocumentChunk]:
        """Find parts of documents relevant to the query."""
        query_embedding = self.encoder.embed_query(query)

        results = self.vectorstore.search(query_embedding, limit)

        return results

    def query_with_faq(self, query: str, limit: int = 3, faq_threshold: int = .95) \
            -> tuple[list[RetrievedDocumentChunk], AnswerType]:
        """If similar question is present in faq return its answer otherwise return relevant parts of documents."""
        query_embedding = self.encoder.embed_query(query)

        try:
            similar_question = self.vectorstore_faq.search(query_embedding, 1)[0]
            if similar_question.score >= faq_threshold:
                return [similar_question], AnswerType.SIMILAR_QUESTION
        except IndexError:
            # Faq vectorstore is empty
            pass

        # fall back to simple query method
        results = self.vectorstore.search(query_embedding, limit)
        return results, AnswerType.RELEVANT_CHUNKS

    def find_best_answer(self, query: str, faq_threshold: int = .95) -> str:
        """Find best answer and return it as string."""
        answer, answer_type = self.query_with_faq(query, faq_threshold=faq_threshold)

        if answer_type == AnswerType.SIMILAR_QUESTION:
            return answer[0].metadata['answer']
        return answer[0].text
