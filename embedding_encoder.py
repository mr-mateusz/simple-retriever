from collections.abc import Sequence
from typing import Iterable

import torch
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


def cls_pooling(model_output: BaseModelOutputWithPoolingAndCrossAttentions) -> torch.Tensor:
    """For each element in batch take embedding vector corresponding to [CLS] token."""
    return model_output['last_hidden_state'][:, 0]


def normalize(t: torch.Tensor) -> torch.Tensor:
    """L2 vector normalization.

    >>> normalize(torch.tensor([[3., 4.]]))
    tensor([[0.6000, 0.8000]])
    """
    norm = t.norm(p=2, dim=1, keepdim=True)
    return t.div(norm)


class Encoder:
    """Class responsible for the creation of embedding vectors for queries and document chunks."""

    # These values are hardcoded here, because the class contains logic that is specific to this model
    # (query prefix, title text separator, pooling) and won't work with other sentence bert models.
    model_checkpoint = 'ipipan/silver-retriever-base-v1.1'
    vector_size = 768

    def __init__(self, normalize: bool = True) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        self.model = AutoModel.from_pretrained(self.model_checkpoint)

        self.normalize = normalize

    def _embed(self, texts: Iterable[str]) -> list[list[float]]:
        # Todo handle too long texts
        tokens = self.tokenizer(texts, padding=True, return_tensors='pt')

        with torch.no_grad():
            model_output = self.model(**tokens)

        embeddings = cls_pooling(model_output)

        if self.normalize:
            embeddings = normalize(embeddings)

        embeddings = embeddings.tolist()

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        text = 'Pytanie: ' + text

        return self._embed([text])[0]

    def embed_documents(self, texts: Sequence[str], titles: Sequence[str] | None = None) -> list[list[float]]:
        if titles:
            if len(texts) != len(titles):
                raise ValueError()

            texts = [f'{title}</s>{text}' for title, text in zip(titles, texts)]

        return self._embed(texts)
