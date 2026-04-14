"""
Shared configuration for the CAM-Peru NLP pipeline.

Every hyperparameter, label mapping, and category schema used across the
pipeline is defined here so that downstream modules do not duplicate constants.
If you change any of these values, the README and ``docs/methodology.md``
should be updated accordingly.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# File-system layout                                                          #
# --------------------------------------------------------------------------- #
# The repository root is one parent up from this file in the flat layout
# (cam_peru/config.py -> repo root).
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = REPO_ROOT / "data"
NOTEBOOK_DIR: Path = REPO_ROOT / "notebook"

# --------------------------------------------------------------------------- #
# Azure OpenAI — read from environment, NEVER hardcode credentials.           #
# --------------------------------------------------------------------------- #
# See .env.example for the full list of expected variables.
AZURE_ENV_VARS = {
    "endpoint":       "ENDPOINT_URL",
    "api_key":        "AZURE_OPENAI_API_KEY",
    "deployment":     "DEPLOYMENT_NAME",
    "api_version":    "API_VERSION",
}
DEFAULT_DEPLOYMENT = "gpt-4o"
DEFAULT_API_VERSION = "2024-05-01-preview"

# --------------------------------------------------------------------------- #
# Classification schema (10 categories, Spanish)                              #
# --------------------------------------------------------------------------- #
# The numeric codes are the canonical identifiers used throughout the pipeline.
# Category 10 ("Razones imposibles de categorizar") is the *Miscellaneous*
# catch-all and is handled explicitly at every downstream step — see
# ``notebook/results.ipynb`` and ``docs/methodology.md``.
CATEGORIES_ES: dict[int, str] = {
    1:  "Creencias paranormales",
    2:  "Desconfianza en la medicina convencional",
    3:  "Uso de productos naturales",
    4:  "Beneficios personales y efectividad clínica",
    5:  "Testimonios y experiencias personales",
    6:  "Tradicionalismo",
    7:  "Accesibilidad y comodidad",
    8:  "Credenciales científicas y profesionales",
    9:  "Ausencia de efectos secundarios",
    10: "Razones imposibles de categorizar",
}

# Spanish → English labels used in every published figure.
CLASS_MAP: dict[str, str] = {
    "Creencias paranormales":                      "Paranormal beliefs",
    "Desconfianza en la medicina convencional":    "Distrust in conventional medicine",
    "Uso de productos naturales":                  "Use of natural products",
    "Beneficios personales y efectividad clínica": "Benefits and effectiveness",
    "Testimonios y experiencias personales":       "Testimonies and personal experiences",
    "Tradicionalismo":                             "Traditionalism",
    "Accesibilidad y comodidad":                   "Comfort and accessibility",
    "Credenciales científicas y profesionales":    "Professional and scientific credentials",
    "Ausencia de efectos secundarios":             "Lack of side effects",
    "Razones imposibles de categorizar":           "Miscellaneous",
}

# Code → English label convenience map.
CODE_TO_EN: dict[int, str] = {
    code: CLASS_MAP[CATEGORIES_ES[code]] for code in CATEGORIES_ES
}

MISCELLANEOUS_LABEL_ES = "Razones imposibles de categorizar"
MISCELLANEOUS_LABEL_EN = "Miscellaneous"
MISCELLANEOUS_CODE = 10

# --------------------------------------------------------------------------- #
# Technique labels (Spanish raw → English display with Trad/Alt prefix)       #
# --------------------------------------------------------------------------- #
TECH_MAP: dict[str, str] = {
    "Pasada de huevo":            "(Trad) Egg cleanse",
    "Huesero":                    "(Trad) Bone healer",
    "Pasa o sobada con alumbres": "(Trad) Alum rub",
    "Uso de barro o arcilla":     "(Trad) Use of mud or clay",
    "Limpia por curanderos":      "(Trad) Healer cleanse",
    "Sobada de cuy":              "(Trad) Guinea pig rub",
    "Pago con coca":              "(Trad) Coca payment",
    "Baños de florecimiento":     "(Trad) Flowering baths",
    "Musicoterapia":              "(Alt) Music therapy",
    "Reflexología":               "(Alt) Reflexology",
    "Quiropraxia":                "(Alt) Chiropractic",
    "Acupuntura":                 "(Alt) Acupuncture",
    "Hidroterapia":               "(Alt) Hydrotherapy",
    "Homeopatía":                 "(Alt) Homeopathy",
    "Reiki":                      "(Alt) Reiki",
    "Trofoterapia":               "(Alt) Trophotherapy",
}

# --------------------------------------------------------------------------- #
# Reproducibility hyperparameters                                             #
# --------------------------------------------------------------------------- #
# These values match the ones used to produce the figures in the manuscript.
# Any change should be accompanied by a documented re-run.

# LLM classification
CLASSIFICATION_SAMPLE_SIZE = 300
CLASSIFICATION_RANDOM_STATE = 42
CLASSIFICATION_TEMPERATURE = 0.0
CLASSIFICATION_TOP_P = 1.0
CLASSIFICATION_MAX_TOKENS = 150
CLASSIFICATION_SEED = 42

# Embeddings
EMBEDDING_MODEL = "distilbert-base-multilingual-cased"
EMBEDDING_MAX_LENGTH = 512

# UMAP
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 2
UMAP_RANDOM_STATE = 42
UMAP_METRIC = "euclidean"

# HDBSCAN
HDBSCAN_MIN_CLUSTER_SIZE = 20
HDBSCAN_MIN_SAMPLES = 1
HDBSCAN_METRIC = "euclidean"
HDBSCAN_CLUSTER_SELECTION_METHOD = "eom"

# Semantic-map figure (Figure 1 of the manuscript)
# -----------------------------------------------------------------------------
# The semantic-map uses a slightly different UMAP configuration than the
# canonical 2-D visualisation: UMAP reduces to 10-D to *drive* the HDBSCAN
# noise filter, and t-SNE produces the 2-D / 3-D projection actually plotted.
# The random state that reproduces the published figure is 55 (not 42).
SEMANTIC_MAP_UMAP_N_COMPONENTS = 10
SEMANTIC_MAP_UMAP_N_NEIGHBORS = 15
SEMANTIC_MAP_TSNE_PERPLEXITY = 30
SEMANTIC_MAP_TSNE_N_ITER = 1000
SEMANTIC_MAP_RANDOM_STATE = 55
SEMANTIC_MAP_DIM_MAJORITY_ALPHA = 0.3   # fade the "Benefits" majority class

# Network analysis
SIMILARITY_THRESHOLD = 0.9     # cosine similarity cutoff for edge inclusion
NETWORK_LAYOUT_SEED = 585
NETWORK_LAYOUT_K = 1.0

# Hierarchical clustering
LINKAGE_METHOD = "average"
DISTANCE_METRIC = "cosine"     # 1 - cosine similarity
