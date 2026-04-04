"""Embedding similarity and LLM-based resume scoring."""

from scoring.embedder import embed_text, similarity_scores
from scoring.llm_scorer import (
    check_lm_studio_connection,
    fetch_loaded_model_id,
    score_candidate_resume,
)

__all__ = [
    "embed_text",
    "similarity_scores",
    "check_lm_studio_connection",
    "fetch_loaded_model_id",
    "score_candidate_resume",
]
