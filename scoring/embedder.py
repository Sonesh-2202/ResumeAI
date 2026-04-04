"""
Sentence-transformer embeddings and cosine similarity vs. job description.
"""

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row of a 2D array for stable cosine similarity.

    Args:
        mat: Array of shape (n, d).

    Returns:
        Row-normalized array; zero rows stay zero.
    """
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    out = (mat / norms).astype(np.float32)
    zero_mask = (norms.flatten() < 1e-11)
    if np.any(zero_mask):
        out[zero_mask] = 0.0
    return out


def load_embedder() -> SentenceTransformer:
    """
    Load the sentence-transformers model (use with st.cache_resource in app).

    Returns:
        SentenceTransformer instance.
    """
    return SentenceTransformer(MODEL_NAME)


def embed_text(model: SentenceTransformer, text: str) -> np.ndarray:
    """
    Embed a single piece of text into a normalized vector.

    Args:
        model: Loaded SentenceTransformer.
        text: Input text (empty string yields zero vector of model dim).

    Returns:
        2D array of shape (1, dim) for cosine_similarity API.
    """
    if not text or not str(text).strip():
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((1, dim), dtype=np.float32)
    vec = model.encode(
        [text.strip()],
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    arr = np.asarray(vec, dtype=np.float32)
    return _l2_normalize_rows(arr)


def similarity_scores(
    model: SentenceTransformer,
    job_description: str,
    resume_texts: Sequence[str],
) -> list[float]:
    """
    Compute cosine similarity between JD embedding and each resume embedding.

    Args:
        model: Loaded SentenceTransformer.
        job_description: Job description text.
        resume_texts: Sequence of resume bodies in the same order as candidates.

    Returns:
        List of floats in [0, 1] (clamped) for each resume.
    """
    jd_emb = embed_text(model, job_description)
    scores: list[float] = []
    for resume in resume_texts:
        r_emb = embed_text(model, resume)
        sim = cosine_similarity(jd_emb, r_emb)[0][0]
        sim = float(np.clip(sim, 0.0, 1.0))
        scores.append(sim)
    return scores


def batch_embed_resumes(
    model: SentenceTransformer,
    resume_texts: Sequence[str],
) -> np.ndarray:
    """
    Encode multiple resumes in one batch for efficiency.

    Args:
        model: Loaded SentenceTransformer.
        resume_texts: List of resume strings.

    Returns:
        Array of shape (n, dim).
    """
    texts = [t.strip() if t else "" for t in resume_texts]
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)
    raw = np.asarray(
        model.encode(texts, convert_to_numpy=True, show_progress_bar=False),
        dtype=np.float32,
    )
    return _l2_normalize_rows(raw)


def similarity_scores_batched(
    model: SentenceTransformer,
    job_description: str,
    resume_texts: Sequence[str],
) -> list[float]:
    """
    Batched similarity: one JD encode, batch resume encodes.

    Args:
        model: Loaded SentenceTransformer.
        job_description: JD text.
        resume_texts: Resume strings.

    Returns:
        Similarity scores in [0, 1].
    """
    jd_emb = embed_text(model, job_description)
    if not resume_texts:
        return []
    res_mat = batch_embed_resumes(model, resume_texts)
    sims = cosine_similarity(jd_emb, res_mat)[0]
    return [float(np.clip(s, 0.0, 1.0)) for s in sims]
