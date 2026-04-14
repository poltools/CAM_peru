# CAM Peru — NLP analysis of open-ended justifications

This repository contains the NLP pipeline associated with the paper
*"Psychosocial predictors and justification patterns of traditional and
alternative medicine in Peru"*. It classifies open-ended Spanish
justifications into a 10-category schema with an LLM, embeds them with a
multilingual BERT variant, clusters them with UMAP + HDBSCAN, and builds
a rhetorical-proximity network between techniques based on the category
composition of their justifications.

The psychometric side of the study (scales, SEM, factor analyses) lives
on OSF alongside the raw `.sav` dataset and the SPSS syntax. This repo
covers only the text-analysis component; the psychometric and NLP
components are designed to be reproducible independently.

## Pipeline at a glance

```
           raw survey              extraction          classification
         (TradyAlt Fase1.xlsx)   (reason lists)      (micro-reason → cat)

 N = 805   ─────────────────►   N = 801 responses ─► 1 517 unique
 completed  │   Azure GPT-4o   │  2 609 reasons  │    classified
 responses  │   extraction     │  (after human-  │    micro-reasons
            │   (temp = 0)     │   in-the-loop   │    (multi-label,
            │                  │   curation)     │    10 categories)
```

- **Stage 1 (Reason extraction).** An Azure OpenAI GPT-4o deployment
  reads each open-ended Spanish response and returns a list of
  nominalised reasons. Four of the 805 completed responses yielded no
  parseable reason list and were dropped; the remaining 801 responses
  were hand-curated by domain experts.
- **Stage 2 (Micro-classification).** Each of the 1 517 unique
  nominalised reasons is tagged with one or more of the 10 categories
  below by the same LLM under a zero-shot prompt.
- **Stage 3 (Document-level aggregation).** Per-respondent category
  codes are produced by rolling up the micro-reason labels back to the
  original document.
- **Stage 4 (Downstream figures).** Embeddings (DistilBERT → UMAP →
  HDBSCAN), the rhetorical-proximity network, and the
  hierarchical-clustering dendrogram are produced inside
  `notebook/results.ipynb` from the aggregate composition tables.

Two human annotators independently coded a random sample of 300
reasons per round (two rounds, see `notebook/annotations.ipynb`) to
compute Krippendorff's α and per-class F1 between raters and against
the LLM output.

## Repository layout

```
CAM_peru/
├── LICENSE
├── README.md                 you are here
├── requirements.txt
├── setup.py
├── .env.example              copy to `.env` and fill in Azure credentials
├── data/
│   └── README.md             pointer to the OSF-hosted derived artifacts
├── docs/
│   └── methodology.md        NLP methodology writeup
├── figures/                  rasterised copies of the published figures
│   └── README.md             figure-to-paper mapping
├── notebook/
│   ├── annotations.ipynb     inter-rater agreement (Krippendorff + F1)
│   └── results.ipynb         final figures
└── cam_peru/                 installable Python package
    ├── config.py             schema, label maps, hyperparameters
    ├── llm_client.py         Azure OpenAI wrapper (env-var credentials)
    ├── classification.py     prompt templates + category runner
    ├── process_text.py       Spanish lemmatisation + stop-words
    ├── embeddings.py         DistilBERT + UMAP + HDBSCAN
    ├── semantic_map.py       Figure-1 helper (UMAP → HDBSCAN → t-SNE plot)
    ├── word_clouds.py        bilingual word clouds
    ├── extract_arguments.py        CLI · reason extraction
    ├── run_classification.py       CLI · document-level classification
    └── run_classification_micro.py CLI · micro-reason classification
```

## Quick-start reproduction

The four pipeline stages above are independently reproducible. Each
stage can be rerun in isolation against the corresponding intermediate
artifact hosted on OSF (see *Data availability*). Stages 1–3 require
Azure credentials and the raw Excel file; stage 4 runs from the
OSF-hosted CSVs alone.

### Requirements

- **Python 3.10–3.12** (tested on 3.12.7). Python 3.13 and 3.14 are
  not yet supported by the pinned `blis`/`thinc` wheels used by spaCy.
  The repository ships a `.python-version` file pinning `3.12.7`; if
  you use [`pyenv`](https://github.com/pyenv/pyenv), the correct
  interpreter is auto-selected as soon as you `cd` into the folder
  (after `pyenv install 3.12.7`). Otherwise install a 3.12.x
  interpreter manually and expose it as `python3` or pass it
  explicitly: `PYTHON_BIN=/path/to/python3.12 bash reproduce.sh`.
- A C toolchain (`gcc`/`clang`) — most dependencies ship wheels, but
  `hdbscan` compiles from source on fresh platforms.

### 1 · Install (one-shot)

The `reproduce.sh` helper at the repo root creates a fresh virtual
environment, installs everything from `requirements.txt`, installs the
package in editable mode, and downloads the spaCy Spanish model + NLTK
Spanish stopwords. Run it once after cloning:

```bash
git clone https://github.com/poltools/CAM_peru.git
cd CAM_peru
bash reproduce.sh
source .venv/bin/activate          # once per shell
```

`reproduce.sh` aborts with a clear message if the active Python is
outside the supported range.

If you prefer to do it manually (make sure `python` resolves to 3.12):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python -m spacy download es_core_news_md
python -c "import nltk; nltk.download('stopwords')"
```

### 2 · (Optional) Configure Azure OpenAI

Only needed if you want to *regenerate* the classifications from raw
text. All downstream analyses can be reproduced from the OSF-hosted
CSVs without any API credentials.

```bash
cp .env.example .env
$EDITOR .env   # fill in ENDPOINT_URL and AZURE_OPENAI_API_KEY
```

### 3 · Run the stages you care about

| Stage | Command | Reads | Writes |
| --- | --- | --- | --- |
| Reason extraction | `python -m cam_peru.extract_arguments --input <source.xlsx> --output data/extracted_reasons.csv` | raw open-ended responses | nominalised reasons |
| Document classification | `python -m cam_peru.run_classification --input <source.xlsx> --output data/classified_reasons.csv` | raw open-ended responses | per-document category codes |
| Micro-classification | `python -m cam_peru.run_classification_micro --input data/arguments.csv --output data/classified_micro_reasons.csv` | extracted nominalised reasons | one-category-per-reason labels |
| Embeddings (semantic map) | `import cam_peru.semantic_map` — see `notebook/results.ipynb` | `document_category_labels.csv` + raw text (local only) | UMAP/t-SNE coordinates + HDBSCAN labels (kept in-memory in the notebook cell outputs; not written to disk) |
| Network / hierarchical clustering | inline in `notebook/results.ipynb` (hyperparameters pinned in `cam_peru/config.py`) | `technique_category_composition.csv` | NetworkX graph + dendrogram figures (rendered inline in the notebook) |

The raw source Excel file (`TradyAlt Fase1.xlsx`) is **not**
redistributed; see [Data availability](#data-availability) below.

## Classification schema

The canonical 10-category schema is defined in
[`cam_peru/config.py`](cam_peru/config.py). Codes 1–9 are the
nine substantive categories; code 10 is the `Razones imposibles de
categorizar` ("Miscellaneous") catch-all. Each document can carry any
non-empty subset of codes (multi-label).

| Code | Spanish | English |
| ---: | --- | --- |
| 1 | Creencias paranormales | Paranormal beliefs |
| 2 | Desconfianza en la medicina convencional | Distrust in conventional medicine |
| 3 | Uso de productos naturales | Use of natural products |
| 4 | Beneficios personales y efectividad clínica | Benefits and effectiveness |
| 5 | Testimonios y experiencias personales | Testimonies and personal experiences |
| 6 | Tradicionalismo | Traditionalism |
| 7 | Accesibilidad y comodidad | Comfort and accessibility |
| 8 | Credenciales científicas y profesionales | Professional and scientific credentials |
| 9 | Ausencia de efectos secundarios | Lack of side effects |
| 10 | Razones imposibles de categorizar | Miscellaneous |

The full category definitions (in Spanish, as consumed by the classifier
prompt) live in
[`cam_peru/classification.py`](cam_peru/classification.py) and
[`docs/methodology.md`](docs/methodology.md).

## Inter-rater agreement

Two domain experts independently annotated random samples of 300
micro-reasons per round using the same 10-category schema. The two
rounds (`review/round_1.xlsx`, `review/round_2.xlsx` on OSF) were run
sequentially: round 1 informed a refinement of the classifier prompt;
round 2 validated the refined pipeline. Each file shares the schema
`{text, annotator_1, annotator_2, consensus, Modelo}` where `Modelo`
is the LLM's label under the same prompt setup and `consensus` is the
adjudicated label after reconciliation (7 rows in round 2 are genuine
unresolved disagreements and are excluded from model-vs-consensus
metrics).

[`notebook/annotations.ipynb`](notebook/annotations.ipynb) reproduces
the numbers cited in the paper's inter-rater reliability section and
in Supplementary Figure S1. Its structure:

| Section | Output |
| --- | --- |
| §2 Inter-annotator reliability | Krippendorff's α between the two human annotators (per round, per category) |
| §3 Model vs. consensus | per-class F1 and α between the LLM and the adjudicated consensus |
| §4 Confusion matrices | row-normalised confusion matrices per round (Supplementary Figure S1) |
| §5 Summary table | the summary table of agreement metrics reported in the paper |

The notebook runs in seconds and needs no Azure credentials — only the
two xlsx files from OSF, placed at `data/review/round_{1,2}.xlsx`.

## Handling of the Miscellaneous (code 10) category

The per-class F1 of Miscellaneous is low (0.14–0.27 in the inter-rater
agreement analysis — see `notebook/annotations.ipynb`). For the
**classification** step it is retained so that human-annotated and
LLM-produced agreement metrics remain directly comparable. For the
**rhetorical-proximity network** and **similarity-based hierarchical
clustering** we exclude code 10 from the composition vectors before
computing cosine similarity — the label does not describe a coherent
semantic direction and including it would compress the effective
vocabulary of substantive reasons. The relevant cells in
`notebook/results.ipynb` drop code 10 from the composition matrix
before computing cosine similarity; the sensitivity analysis that
keeps code 10 is documented in `docs/methodology.md`.

## Pipeline hyperparameters

All hyperparameters are centralised in
[`cam_peru/config.py`](cam_peru/config.py):

- **LLM classification** — Azure OpenAI GPT-4o deployment, temperature 0.0,
  top-p 1.0, seed 42, max tokens 150, sample size 300, random state 42.
- **Embeddings** — `distilbert-base-multilingual-cased`, mean-pooled over
  attention-masked token states, max length 512.
- **UMAP** — `n_neighbors=15`, `n_components=2`, `random_state=42`,
  `metric="euclidean"`.
- **HDBSCAN** — `min_cluster_size=20`, `min_samples=1`, `metric="euclidean"`,
  `cluster_selection_method="eom"`.
- **Network** — cosine similarity on scaled composition vectors,
  threshold 0.9, NetworkX `spring_layout` with `seed=585`, `k=1.0`.
- **Hierarchical clustering** — cosine distance, `linkage(method="average")`.

Silhouette scores are reported on non-noise points only (HDBSCAN convention).

## Paper figures → notebook cells

Rasterised copies of every published figure are included under
[`figures/`](figures/) so readers can inspect the results without
rerunning the notebooks. Every figure is also produced by a
clearly-headed cell in one of the two notebooks:

| Figure / table in the paper | Notebook | Section header |
| --- | --- | --- |
| Figure 1 — semantic clustering of reasons (DistilBERT → UMAP → HDBSCAN → t-SNE) | `notebook/results.ipynb` | *Semantic map (Figure 1)* |
| Figure 2 — normalised distribution of categories per technique type | `notebook/results.ipynb` | *Distribution of reasons per therapy type* (composition bar chart) |
| Figure 3 — pairwise co-occurrence of categories within justifications | `notebook/results.ipynb` | *Distribution of reasons per therapy type* (cosine-similarity / co-occurrence cells) |
| Figure 4 — similarity network (cosine ≥ 0.9, `spring_layout` seed 585) | `notebook/results.ipynb` | *Distribution of reasons per therapy type* (NetworkX cells at the end of the section) |
| Word-cloud figures in the SI | `notebook/results.ipynb` | *Basic inspection: word clouds* / *wordcloud by technique* |
| Supplementary Figure S1 — normalised confusion matrices (IRR) | `notebook/annotations.ipynb` | *§4 Supplementary Figure S1 — normalised confusion matrices* |
| Inter-rater reliability summary table | `notebook/annotations.ipynb` | *§5 Summary table* |

Figures 2–4 currently share the single *Distribution of reasons per
therapy type* heading; the constituent cells are contiguous but
unmarked. If you want to rerun just one of those figures, the relevant
cells are adjacent and clearly named by their `matplotlib` /
`networkx` calls.

## Data availability

The derived artifacts needed to reproduce every downstream analysis in
this repository are hosted on OSF alongside the psychometric data:

- `classified_micro_reasons.csv` — 1 517-row mapping from LLM-extracted
  nominalised reasons to category labels. Contains paraphrased
  nominalisations (e.g. "eficacia percibida"), not verbatim respondent
  text.
- `document_category_labels.csv` — per-document category codes and
  English labels for all 801 documents, joined with technique and
  technique type. Raw response text is intentionally omitted.
- `technique_category_composition.csv` and
  `technique_type_category_composition.csv` — aggregate
  technique × category count tables sufficient to reproduce all
  similarity-network and clustering figures.
- `CAM_translations.csv` — Spanish → English override map for the
  word-cloud pipeline.
- `review/round_1.xlsx` and `review/round_2.xlsx` — two-annotator
  evaluation samples (300 rows each) used to compute inter-rater
  reliability and model-vs-consensus F1 in
  `notebook/annotations.ipynb`. Schema
  `{text, annotator_1, annotator_2, consensus, Modelo}`. Texts are
  paraphrased nominalised reasons, not verbatim respondent utterances.

The raw open-ended responses (`TradyAlt Fase1.xlsx` column `Abierta`)
are **not** redistributed. They are covered by the study's IRB consent
and are available on reasonable request from the corresponding author.

See [`data/README.md`](data/README.md) for the OSF DOI and column-level
documentation. Once downloaded, place the CSVs into `data/` to reproduce
the notebooks.

## License

MIT — see [`LICENSE`](LICENSE).

## Citation

If you use this code, please cite the preprint:

> Fasce, A., Lopez-Lopez, E., Rodríguez-Ferreiro, J., Rosales-Trabuco, J. E.,
> & Barberia, I. (2026, April 14). *Prevalence of traditional and alternative
> medicine in Peru, and impact of empathetic refutations on participant's
> endorsement*. https://doi.org/10.17605/OSF.IO/CXW5U

This citation will be updated once the paper is published in the peer-reviewed journal.
