"""Hybrid search: finds the passages most relevant to a question.

Hybrid means two search methods run together and their results are
combined into one ranked list:

- Vector search compares meanings. The question is turned into an
  embedding (a list of numbers) and Weaviate finds chunks whose
  embeddings are closest, even when they use different words.
- Keyword search (called BM25) matches the actual words, which is
  exact and reliable for names, codes and specific terms.

The alpha setting balances the two: 0.0 is keyword only, 1.0 is
vector only and 0.5 weighs them equally. Weaviate runs both and
merges the scores.
"""

from dataclasses import dataclass

import weaviate.classes as wvc

from src.config import settings


@dataclass
class RetrievedPassage:
    """One search result: the text, where it came from and its score."""

    text: str
    source: str
    score: float


class Retriever:
    """Runs hybrid searches against the Weaviate collection."""

    def __init__(self, client, embedder):
        self._collection = client.collections.get(settings.collection_name)
        self._embedder = embedder

    def search(
        self,
        question: str,
        top_k: int | None = None,
        alpha: float | None = None,
    ) -> list[RetrievedPassage]:
        """Returns the best matching passages for the question."""
        top_k = top_k or settings.top_k
        alpha = settings.hybrid_alpha if alpha is None else alpha

        # The question gets the same embedding treatment as the chunks
        # did at loading time, so meanings can be compared like for like.
        question_vector = self._embedder.encode(question).tolist()

        response = self._collection.query.hybrid(
            query=question,              # used by the keyword (BM25) half
            vector=question_vector,      # used by the vector half
            alpha=alpha,                 # balance between the two halves
            limit=top_k,
            return_metadata=wvc.query.MetadataQuery(score=True),
        )

        return [
            RetrievedPassage(
                text=obj.properties.get("text", ""),
                source=obj.properties.get("source", "unknown"),
                score=float(obj.metadata.score or 0.0),
            )
            for obj in response.objects
        ]
