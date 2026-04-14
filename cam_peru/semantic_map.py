"""
Semantic map — Figure 1 of the manuscript.

This module reproduces the 2-D + 3-D panel figure that visualises the
semantic landscape of the individual (micro) justifications:

    DistilBERT multilingual embeddings
        └── UMAP to 10-D (``cluster_space``)
            └── HDBSCAN noise filter (drops label == -1)
                └── t-SNE to 2-D / 3-D (``plot_space``)
                    └── Scatter coloured by taxonomic category
                         (as assigned by the LLM classifier).

The HDBSCAN step is used **only to drop noise points**, not to colour the
scatter. Coloring comes from the independently-assigned taxonomic
category, which is the whole point of Figure 1: unsupervised geometry vs.
expert-derived taxonomy.

Hyperparameters live in :mod:`cam_peru.config` under the
``SEMANTIC_MAP_*`` prefix and match the values used for the published
figure (``random_state = 55``).

Typical use from a notebook::

    from cam_peru.semantic_map import compute_semantic_map, plot_semantic_map

    df_2d, df_3d = compute_semantic_map(
        texts=micro["text"],
        categories_es=micro["categoria_single"],
    )
    plot_semantic_map(df_2d, df_3d, out_path="figures/figure_1.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import hdbscan
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib import font_manager, gridspec
from sklearn.manifold import TSNE

from .config import (
    CLASS_MAP,
    SEMANTIC_MAP_DIM_MAJORITY_ALPHA,
    SEMANTIC_MAP_RANDOM_STATE,
    SEMANTIC_MAP_TSNE_N_ITER,
    SEMANTIC_MAP_TSNE_PERPLEXITY,
    SEMANTIC_MAP_UMAP_N_COMPONENTS,
    SEMANTIC_MAP_UMAP_N_NEIGHBORS,
)
from .embeddings import compute_distilbert_embeddings


# --------------------------------------------------------------------------- #
# Core pipeline                                                               #
# --------------------------------------------------------------------------- #


def _umap_hdbscan_noise_mask(
    embeddings: np.ndarray, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (cluster_labels, valid_mask) using UMAP→HDBSCAN.

    ``valid_mask`` is True where HDBSCAN did NOT flag the point as noise.
    """
    reducer = umap.UMAP(
        n_neighbors=SEMANTIC_MAP_UMAP_N_NEIGHBORS,
        n_components=SEMANTIC_MAP_UMAP_N_COMPONENTS,
        random_state=random_state,
    )
    cluster_coords = reducer.fit_transform(embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=20, min_samples=1, metric="euclidean"
    )
    labels = clusterer.fit_predict(cluster_coords)
    return labels, labels != -1


def _tsne(embeddings: np.ndarray, dim: int, random_state: int) -> np.ndarray:
    return TSNE(
        n_components=dim,
        perplexity=SEMANTIC_MAP_TSNE_PERPLEXITY,
        n_iter=SEMANTIC_MAP_TSNE_N_ITER,
        random_state=random_state,
    ).fit_transform(embeddings)


def compute_semantic_map(
    texts: Iterable[str],
    categories_es: Iterable[str],
    *,
    embeddings: np.ndarray | None = None,
    random_state: int = SEMANTIC_MAP_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the 2-D and 3-D t-SNE projections used in Figure 1.

    Parameters
    ----------
    texts
        Reason-level Spanish text strings. Ignored if ``embeddings`` is
        supplied, but still used to populate the returned data frame.
    categories_es
        Spanish category label per reason (single-label). Used for
        colouring; the 2-D frame drops HDBSCAN noise points, the 3-D
        frame retains them.
    embeddings
        Optional precomputed ``(N, 768)`` sentence embeddings. If
        omitted, :func:`cam_peru.embeddings.compute_distilbert_embeddings`
        is invoked on ``texts``.
    random_state
        Seed shared by UMAP and both t-SNE runs. Defaults to
        ``SEMANTIC_MAP_RANDOM_STATE`` (55), which reproduces the published
        figure.

    Returns
    -------
    (df_2d, df_3d)
        Data frames with columns ``text``, ``categoria_es``, ``cluster``,
        and ``dim1``/``dim2``/``dim3`` as appropriate. ``df_2d`` has
        HDBSCAN noise filtered out; ``df_3d`` retains all points.
    """
    texts = list(texts)
    categories_es = list(categories_es)
    if len(texts) != len(categories_es):
        raise ValueError("texts and categories_es must have the same length")

    if embeddings is None:
        embeddings = compute_distilbert_embeddings(texts)
    embeddings = np.asarray(embeddings)
    if embeddings.shape[0] != len(texts):
        raise ValueError(
            f"embeddings has {embeddings.shape[0]} rows but texts has {len(texts)}"
        )

    cluster_labels, valid = _umap_hdbscan_noise_mask(embeddings, random_state)

    tsne_2d = _tsne(embeddings, dim=2, random_state=random_state)
    tsne_3d = _tsne(embeddings, dim=3, random_state=random_state)

    base = pd.DataFrame(
        {
            "text": texts,
            "categoria_es": categories_es,
            "cluster": cluster_labels,
        }
    )

    df_2d = base.assign(dim1=tsne_2d[:, 0], dim2=tsne_2d[:, 1]).loc[valid].reset_index(drop=True)
    df_3d = base.assign(dim1=tsne_3d[:, 0], dim2=tsne_3d[:, 1], dim3=tsne_3d[:, 2]).reset_index(drop=True)

    return df_2d, df_3d


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #


def _build_color_map(
    categories: pd.Series, palette: str, dim_majority_alpha: float
) -> dict[str, tuple[float, float, float, float]]:
    unique = sorted(categories.unique())
    cmap = plt.cm.get_cmap(palette, len(unique))
    val_to_color = {val: cmap(i) for i, val in enumerate(unique)}

    if dim_majority_alpha < 1.0 and len(categories):
        majority = categories.value_counts().idxmax()
        base = np.array(val_to_color[majority][:3])
        dimmed = base * dim_majority_alpha + (1 - dim_majority_alpha)
        val_to_color[majority] = (*np.clip(dimmed, 0, 1), 1.0)

    return val_to_color


def plot_semantic_map(
    df_2d: pd.DataFrame,
    df_3d: pd.DataFrame,
    *,
    out_path: str | Path | None = None,
    palette: str = "tab10",
    dim_majority_alpha: float = SEMANTIC_MAP_DIM_MAJORITY_ALPHA,
    class_map: dict[str, str] | None = None,
    dpi: int = 300,
) -> plt.Figure:
    """Render the 2-D / 3-D side-by-side scatter with a shared bottom legend.

    Mirrors the ``cluster_visual_bottom_legend_final`` figure used in the
    manuscript. The ``class_map`` translates Spanish category labels to
    English for the legend (defaults to :data:`cam_peru.config.CLASS_MAP`).
    """
    class_map = class_map or CLASS_MAP
    val_to_color = _build_color_map(
        df_2d["categoria_es"], palette=palette, dim_majority_alpha=dim_majority_alpha
    )

    fig = plt.figure(figsize=(18, 9))
    gs = gridspec.GridSpec(nrows=2, ncols=5, height_ratios=[20, 1])

    # --- 2-D panel ------------------------------------------------------- #
    ax1 = fig.add_subplot(gs[0, 0:2])
    for val in sorted(df_2d["categoria_es"].unique()):
        sub = df_2d[df_2d["categoria_es"] == val]
        ax1.scatter(
            sub["dim1"], sub["dim2"],
            c=[val_to_color[val]],
            s=50, alpha=0.85,
            edgecolors="k", linewidths=0.2,
        )
    ax1.set_aspect("equal")
    ax1.grid(True, linestyle=":", alpha=0.4)
    ax1.set_xlabel("Dimension 1", fontsize=12)
    ax1.set_ylabel("Dimension 2", fontsize=12)

    # --- 3-D panel ------------------------------------------------------- #
    ax2 = fig.add_subplot(gs[0, 2:], projection="3d")
    for val in sorted(df_3d["categoria_es"].unique()):
        sub = df_3d[df_3d["categoria_es"] == val]
        color = val_to_color.get(val, val_to_color[sorted(df_2d["categoria_es"].unique())[0]])
        ax2.scatter(
            sub["dim1"], sub["dim2"], sub["dim3"],
            c=[color],
            s=50, alpha=0.6,
            edgecolors="k", linewidths=0.2,
        )
    ax2.view_init(elev=30, azim=-45)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_zticks([])
    ax2.set_xlabel("X", fontsize=10)
    ax2.set_ylabel("Y", fontsize=10)
    ax2.set_zlabel("Z", fontsize=10)

    # --- shared bottom legend ------------------------------------------- #
    handles = [
        mpatches.Patch(color=val_to_color[val], label=class_map.get(val, val))
        for val in sorted(df_2d["categoria_es"].unique())
    ]
    legend_ax = fig.add_subplot(gs[1, :])
    legend_ax.axis("off")
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=5,
        frameon=False,
        borderpad=0.3,
        handletextpad=0.5,
        columnspacing=1.2,
        title="Reason Categories",
        prop=font_manager.FontProperties(size=14),
        title_fontproperties=font_manager.FontProperties(size=15),
    )

    plt.subplots_adjust(hspace=0.05, bottom=0.08, top=0.95)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    return fig
