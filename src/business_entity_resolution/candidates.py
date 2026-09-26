"""Candidate-generation interfaces.

Baseline implementation will combine strict rules and character TF-IDF retrieval.
Dense retrieval is optional and must be evaluated by candidate recall before adoption.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidatePair:
    source1_entity_id: str
    candidate_entity_id: str
    retrieval_method: str
    retrieval_score: float | None = None
    retrieval_rank: int | None = None
