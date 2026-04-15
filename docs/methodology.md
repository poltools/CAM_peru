# NLP methodology

This document describes the text-analysis pipeline used in this
repository. It is intended to be read alongside `README.md` and
`notebook/annotations.ipynb`.

## 1 · Data and preprocessing

### Source
The input is a national survey of N = 805 completed responses. The
extraction pass yields a parseable reason list for 801 of those; after
micro-classification, three documents resolve to *Miscellaneous only*
(code 10) and are excluded from substantive analyses, leaving
**N = 798 documents with at least one substantive reason**. This
matches the figure reported in the manuscript. Respondents named a
CAM technique they considered most effective and justified that choice
in free text. Justifications were given in Peruvian Spanish.

### Preprocessing (for the word-cloud and lexical-frequency figures)
Implemented in `cam_peru/process_text.py`. For each response we:

1. Lower-case and strip non-letter characters (preserving accented
   Spanish characters and the letter `ñ`).
2. Tokenise and lemmatise with spaCy's `es_core_news_md` model.
3. Drop NLTK Spanish stop-words.
4. Drop a hand-curated list of high-frequency function words and
   overly general lemmas (`ser`, `hacer`, `algo`, …).
5. Strip POS tags in `{PRON, DET, ADP, CCONJ, SCONJ, PART, INTJ}`.
6. Drop accents via `unidecode` for deterministic comparison.

Preprocessing is **not** applied before the LLM classification step —
the classifier sees the raw response so that context words remain
available to the prompt.

## 2 · LLM-assisted classification

### Schema
A 10-category schema of attitude roots was developed iteratively by the
psychology team, informed by prior work on pseudoscientific beliefs and
CAM endorsement. Codes 1–9 are substantive and code 10 is the
`Razones imposibles de categorizar` ("Miscellaneous") catch-all (see
`config.py` for the canonical list).

### Two-pass design
The pipeline runs two passes over the raw responses:

1. **Reason extraction** (`cam_peru.extract_arguments`) — GPT-4o
   is prompted to condense each free-text justification into a list of
   nominalised Spanish reasons such as `"eficacia percibida"` or
   `"desconfianza hacia farmacéuticas"`. These nominalisations act as
   an intermediate representation that is (a) safer to share publicly
   than the verbatim respondent voice, and (b) cheaper to classify in
   the second pass.
2. **Micro-classification** (`cam_peru.run_classification_micro`) —
   each nominalised reason is then independently mapped to one
   category code, using the MICRO_REASON_PROMPT template which forces
   the most specific single category. **All per-class F1 and
   Krippendorff's α figures reported in the paper are computed on the
   output of this pass.**

A complementary **document-level classification** pass
(`cam_peru.run_classification`) is also available for multi-label
exploration at the respondent level; it feeds the technique-by-category
composition matrix used in the network analysis but is not the object
of the inter-rater agreement analysis.

### Prompt engineering and determinism
All three prompt templates are stored verbatim in
`cam_peru/classification.py`. We call GPT-4o (`gpt-4o` deployment
on Azure OpenAI, API version `2024-05-01-preview`) with

- `temperature = 0.0`
- `top_p = 1.0`
- `seed = 42`
- `max_tokens = 150`

Even at `temperature = 0`, Azure OpenAI does not guarantee bit-exact
determinism across runs. The published inter-rater agreement numbers
(see §3) are computed on a single classification run whose outputs are
archived in the micro-classification CSV on OSF; re-running the prompt
yields materially equivalent but not byte-identical outputs.

### Handling of parse failures
`cam_peru.classification.parse_category_list` strips Markdown fences
and attempts `ast.literal_eval`. Documents for which no integer code in
`{1, …, 10}` can be recovered are recorded with an empty code list and
excluded from the composition matrices (i.e. they are treated as
missing, not as Miscellaneous).

## 3 · Inter-rater agreement and F1

### Design
Two independent human annotators (referred to here as `annotator_1`
and `annotator_2`) labelled a random sample of **300 nominalised
reasons** against the same 10-category schema in **each of two rounds**.
In round 1 the sample was classified by the LLM using the initial
prompt; round 2 used a refined prompt applied to a fresh 300-reason
sample. After each round, disagreements between the two human
annotators were resolved through discussion to produce a **consensus
label**, which serves as the reference standard for evaluating model
performance.

### Reported metrics
Computed in `notebook/annotations.ipynb`:

- **Inter-annotator reliability** — Krippendorff's α between
  `annotator_1` and `annotator_2`, per round.
- **Model vs. consensus** — per-class precision / recall / F1,
  macro-F1, and Krippendorff's α between the LLM and the human
  consensus label, per round.
- **Confusion matrices** — row-normalised, per round
  (Supplementary Figure S1).

### Headline numbers (from the manuscript)

| Round | Human α | Model-consensus α | Model-consensus α (no Misc) | Weighted F1 |
| --- | ---: | ---: | ---: | ---: |
| 1 (initial prompt)  | 0.67 | 0.61 | 0.85 | 0.80 |
| 2 (refined prompt)  | 0.75 | 0.81 | 0.85 | 0.82 |

### Miscellaneous (code 10)
The Miscellaneous catch-all is kept in the classification metrics so
that the numbers remain directly comparable to the human annotators.
Its per-class F1 is low (**0.27 in round 1, 0.14 in round 2**) because
the category is defined negatively — by the absence of any substantive
reason — so small differences in rater strictness have a large
proportional effect. Code 10 is **retained** in the composition
vectors used for the rhetorical-proximity network (see §5); the
rationale is given there.

## 4 · Sentence embeddings and density-based clustering

Implemented in `cam_peru/embeddings.py`.

1. Each preprocessed reason is tokenised with
   `distilbert-base-multilingual-cased` (max length 512).
2. The final hidden states are mean-pooled, weighted by the attention
   mask, to produce a 768-dimensional sentence vector per reason.
3. **Clustering space.** UMAP reduces the 768-D embeddings to 10-D
   (`n_neighbors=15`, `n_components=10`, `random_state=55`,
   `metric="euclidean"`). HDBSCAN is run on this 10-D manifold with
   `min_cluster_size=20`, `min_samples=1`, `metric="euclidean"`.
4. **Plotting space.** A separate t-SNE projection of the original
   768-D embeddings to 2-D / 3-D (`random_state=55`, perplexity and
   n_iter pinned in `config.py`) is used for the Figure 1 scatter
   only. The 2-D panel drops HDBSCAN noise points (`label == -1`); the
   3-D panel retains all points.
5. **Cluster quality (diagnostic only).** As a diagnostic of the
   unsupervised clustering step, the silhouette score is computed on
   the 10-D UMAP clustering space — the same space HDBSCAN operates
   on — **over non-noise points only** (`label != -1`), matching the
   standard HDBSCAN convention and the original notebook's
   `flexible_cluster_plot` implementation. Running the pipeline on the
   shipped `classified_micro_reasons.csv` at
   `SEMANTIC_MAP_RANDOM_STATE = 55` gives **silhouette ≈ 0.48**
   (1 117 non-noise of 1 517 reasons), matching the value reported by
   the original working notebook (`lab_1_v3.ipynb`, cell 91:
   *Silhouette Score in UMAP space = 0.45*) to within stochastic drift
   across UMAP/sklearn versions. `compute_semantic_map` returns this
   value as its third tuple element; see `notebook/results.ipynb` cell
   *Semantic map (Figure 1)* for the printed number. The silhouette
   rates internal cohesion vs. separation of the HDBSCAN clusters; it
   is **not** a measure of the agreement between the unsupervised
   partition and the LLM-assigned taxonomic categories used to colour
   Figure 1, nor is it computed in the 2-D t-SNE plotting space.

We sampled representative sentences per cluster from the first row in
each cluster (deterministic given a fixed random state) for figure
annotation.

## 5 · Rhetorical-proximity network

Implemented inline in `notebook/results.ipynb`. Hyperparameters
(similarity threshold, spring-layout seed, linkage method) are pinned
in `cam_peru/config.py` so the notebook cells can be rerun without
editing.

### Composition matrix
For each technique we build a row vector over the 10 categories, whose
entries are the number of times that category was assigned to any
respondent who named that technique. Entries are then

1. divided by the grand total of the matrix (global frequency weighting),
   then
2. renormalised so each row sums to 1.

### Miscellaneous (code 10)
Code 10 is **retained** in the composition vectors used for
similarity-based analyses, matching the pipeline that produced the
published Figure 4. Rationale: the entries routed to code 10 are not
all random noise — many are affective states (tranquility, peace,
joy), personal dispositions, or social framings (see Supplementary
Table S7 for representative examples). Because we cannot cleanly
separate noise from signal inside this category without a separate
sub-schema (which we did not attempt), we preferred to retain the
content rather than drop it. Supplementary Tables S6 and S7 make the
composition of each category inspectable: Table S6 lists illustrative
nominalised reasons for the nine substantive categories, Table S7
lists representative reasons that landed in Miscellaneous, grouped
post hoc for readability only. A sensitivity variant that drops code
10 from the composition vectors can be produced by re-running the
notebook cell on the column-subset matrix; see `si_s2/sensitivity.py`
for the comparison script.

### Similarity graph
Pairwise cosine similarity is computed on the scaled composition
vectors (10 categories, including Miscellaneous). An undirected graph is built with one node per
technique; an edge `(i, j)` is added iff `cos_sim(i, j) ≥ 0.9`. Nodes
carry `type` ("Traditional" / "Alternative") and `display` (label
without the prefix) attributes. The published layout uses
`nx.spring_layout(seed=585, k=1.0)`.

### Hierarchical clustering (auxiliary figure)
We also cluster techniques directly by `1 - cosine_similarity`, using
`scipy.cluster.hierarchy.linkage(method="average")` on the condensed
distance matrix. The dendrogram ordering is used to reorder the
heatmap of pairwise distances.

## 6 · Reproducibility summary

| Component | Seed / determinism |
| --- | --- |
| LLM classification | `temperature=0`, `seed=42`, best-effort only (not bit-exact on Azure). |
| UMAP | `random_state=42`. |
| HDBSCAN | Deterministic given embeddings and `min_cluster_size`, `min_samples`. |
| NetworkX layout | `seed=585`. |
| Hierarchical clustering | Deterministic. |

All hyperparameters are in `cam_peru/config.py`; any change there
should be accompanied by a documented re-run and an update to this file.

## 7 · Sample representativeness and limitations

The original survey is a nationally-relevant but non-probabilistic
sample of Peruvian adults recruited online (detailed sampling design in
the main manuscript). The open-ended subcorpus therefore inherits the
same sampling constraints. The NLP pipeline itself does not mitigate
sampling bias: it reports the distribution of justifications *within
the observed sample*, and cross-sectional patterns should not be
interpreted causally.

The question wording elicits a justification for the technique the
respondent considers most *effective*, not necessarily one they
personally use. This distinction is respected throughout the paper
(see §2 of the main text).

Techniques that occupy hybrid cultural positions (acupuncture, reiki)
are discussed in the paper as sitting between the Traditional and
Alternative poles; the network figure in `notebook/results.ipynb`
visualises this explicitly.
