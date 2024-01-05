from dataclasses import dataclass, field

from utils import generate_id


@dataclass
class DocumentChunk:
    text: str
    metadata: dict
    vector: list[float] = field(default_factory=list)
    id: str = field(default_factory=generate_id)


@dataclass
class RetrievedDocumentChunk(DocumentChunk):
    score: float = .0


@dataclass
class FaqEntry:
    question: str
    answer: str
    category: str | None
