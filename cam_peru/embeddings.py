"""
Sentence-embedding and unsupervised-clustering pipeline for the open-ended
CAM justifications.

Pipeline
--------
1. :func:`compute_distilbert_embeddings` — tokenise each (preprocessed) reason
   with ``distilbert-base-multilingual-cased``, mean-pool the token hidden
   states, and return a ``(N, 768)`` array of sentence vectors.
2. :func:`reduce_with_umap` — reduce to 2-D with UMAP for visualisation and
   density-based clustering.
3. :func:`cluster_with_hdbscan` — run HDBSCAN on the reduced space.
4. :func:`silhouette` — evaluate cluster quality, **excluding noise points
   (label == -1) as is conventional for HDBSCAN.**

All hyperparameters are sourced from :mod:`cam_peru.config`.
"""

from __future__ import annotations

from typing import Iterable

import hdbscan
import numpy as np
import torch
import umap
from sklearn.metrics import silhouette_score
from transformers import DistilBertModel, DistilBertTokenizer

from .config import (
    EMBEDDING_MAX_LENGTH,
    EMBEDDING_MODEL,
    HDBSCAN_CLUSTER_SELECTION_METHOD,
    HDBSCAN_METRIC,
    HDBSCAN_MIN_CLUSTER_SIZE,
    HDBSCAN_MIN_SAMPLES,
    UMAP_METRIC,
    UMAP_N_COMPONENTS,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
)

# --------------------------------------------------------------------------- #
# Model singletons — DistilBERT is expensive to reload.                       #
# --------------------------------------------------------------------------- #
_tokenizer: DistilBertTokenizer | None = None
_model: DistilBertModel | None = None


def _get_model() -> tuple[DistilBertTokenizer, DistilBertModel]:
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = DistilBertTokenizer.from_pretrained(EMBEDDING_MODEL)
        _model = DistilBertModel.from_pretrained(EMBEDDING_MODEL)
        _model.eval()
    return _tokenizer, _model


def embed_one(text: str) -> np.ndarray:
    """Return a single ``(768,)`` sentence vector via mean-pooled DistilBERT.

    Mean pooling mirrors the published pipeline. Padding tokens are masked
    implicitly via ``attention_mask`` to keep short and long inputs on the
    same scale.
    """
    tokenizer, model = _get_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=EMBEDDING_MAX_LENGTH,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    token_states = outputs.last_hidden_state          # (1, L, 768)
    mask = inputs["attention_mask"].unsqueeze(-1)     # (1, L, 1)
    summed = (token_states * mask).sum(dim=1)         # (1, 768)
    counts = mask.sum(dim=1).clamp(min=1)             # (1, 1)
    pooled = (summed / counts).squeeze(0).cpu().numpy()
    return pooled


def compute_distilbert_embeddings(texts: Iterable[str]) -> np.ndarray:
    """Compute mean-pooled embeddings for a sequence of strings.

    Returns
    -------
    ndarray of shape ``(N, 768)``.
    """
    return np.vstack([embed_one(t) for t in texts])


def reduce_with_umap(
    embeddings: np.ndarray,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    n_components: int = UMAP_N_COMPONENTS,
    random_state: int = UMAP_RANDOM_STATE,
    metric: str = UMAP_METRIC,
) -> np.ndarray:
    """UMAP dimensionality reduction with published defaults."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        random_state=random_state,
        metric=metric,
    )
    return reducer.fit_transform(embeddings)


def cluster_with_hdbscan(
    reduced: np.ndarray,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
    metric: str = HDBSCAN_METRIC,
    cluster_selection_method: str = HDBSCAN_CLUSTER_SELECTION_METHOD,
) -> np.ndarray:
    """Run HDBSCAN and return an array of integer cluster labels.

    Noise points are labelled ``-1`` — downstream consumers should filter
    them out explicitly (see :func:`silhouette`).
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
    )
    return clusterer.fit_predict(reduced)


def silhouette(reduced: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette score computed over non-noise points.

    Returns ``None`` when fewer than two non-noise clusters are present.
    """
    valid = labels != -1
    if valid.sum() < 2:
        return None
    unique = np.unique(labels[valid])
    if len(unique) < 2:
        return None
    return float(silhouette_score(reduced[valid], labels[valid]))


__all__ = [
    "embed_one",
    "compute_distilbert_embeddings",
    "reduce_with_umap",
    "cluster_with_hdbscan",
    "silhouette",
]
