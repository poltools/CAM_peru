"""
cam_peru: NLP pipeline for the "CAM use in Peru" study.

Submodules
----------
- ``config``          Category schema, label maps, hyperparameters.
- ``process_text``    Spanish text preprocessing (spaCy + NLTK).
- ``llm_client``      Thin wrapper around Azure OpenAI (env-var based).
- ``classification``  Prompt templates and classifier runners.
- ``embeddings``      DistilBERT sentence embeddings + UMAP + HDBSCAN.
- ``semantic_map``    Figure-1 helper: UMAP→HDBSCAN noise filter + t-SNE plot.
- ``word_clouds``     Bilingual word-cloud rendering.

The rhetorical-proximity network (cosine similarity on technique ×
category composition, NetworkX ``spring_layout``) and the
hierarchical-clustering dendrogram are produced inline in
``notebook/results.ipynb`` to keep the figure code co-located with its
output. The hyperparameters are pinned in ``config.py``.

CLI entry points (invoke with ``python -m cam_peru.<name>``)
------------------------------------------------------------
- ``extract_arguments``         Reason-extraction pass.
- ``run_classification``        Document-level classification.
- ``run_classification_micro``  Micro-reason classification.
"""

__version__ = "0.1.0"
