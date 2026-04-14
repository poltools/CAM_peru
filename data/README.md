# `data/` — OSF-hosted derived artifacts

This folder is intentionally empty in the GitHub repository. The derived
NLP artifacts live on the project's OSF page, alongside the psychometric
`.sav` dataset and the SPSS syntax file.

**OSF DOI:** [`10.17605/OSF.IO/CXW5U`](https://doi.org/10.17605/OSF.IO/CXW5U)

Download the CSVs below and place them in this folder to reproduce the
notebooks. No raw verbatim respondent text is distributed; raw
open-ended responses are covered by the study's IRB consent and are
available on reasonable request.

## Files on OSF

### `classified_micro_reasons.csv`
LLM-extracted nominalised reasons, one per row, with their assigned
category. Delimiter `|`. 1 517 rows.

| Column | Type | Description |
| --- | --- | --- |
| `index` | int | Row id of the nominalised reason inside the extraction output. |
| `text` | str | Nominalised reason in Spanish (e.g. `"eficacia percibida"`). Paraphrased — **not** a verbatim participant utterance. |
| `categoria` | str | Python list of category codes, e.g. `"[4]"`. Produced by the micro-classification prompt (see `cam_peru/classification.py`). |
| `categoria_` | str | Python list of Spanish category names, aligned with `categoria`. |

Produced by `python -m cam_peru.run_classification_micro`.

### `document_category_labels.csv`
Per-document category assignments with technique and technique type.
Delimiter `,`. **801 rows** shipped (one per respondent with a
non-empty extracted reason list). Of these, three documents are
classified as *Miscellaneous only* (code 10) and are excluded from the
substantive analyses reported in the manuscript, which therefore cites
**N = 798**. All 801 rows are retained in the CSV for transparency;
notebooks that report substantive-reason analyses drop the three
Misc-only documents before aggregation.

| Column | Type | Description |
| --- | --- | --- |
| `id_` | int | Respondent id. |
| `technique` | str | Technique the respondent named (Spanish, raw). |
| `technique_type` | str | `Traditional` or `Alternative` (mapping in `cam_peru/config.py`). |
| `category_codes` | str | Python list of unique integer codes assigned to that respondent's document. |
| `category_labels_en` | str | Python list of the same labels in English. |

Derivation: for each respondent we take the list of nominalised reasons
produced by the extraction pass, look each one up in
`classified_micro_reasons.csv`, collect the unique categories, and
export. This matches the `arguments_df` / `associated_classes` structure
used in the analysis notebooks. Raw response text is intentionally
dropped.

### `technique_category_composition.csv`
Aggregate `technique × category` count table (long-form labels with
`(Trad)` / `(Alt)` prefixes). 16 techniques × 10 categories. Sufficient
to reproduce all similarity-network, scaled-bar, and
hierarchical-clustering figures without accessing any respondent-level
data.

### `technique_type_category_composition.csv`
Same as above but collapsed to the two-row `Traditional` / `Alternative`
grouping. Used for the auxiliary composition plot.

### `CAM_translations.csv`
Two-column Spanish → English override map for the word-cloud pipeline
(`word`, `translation`). Used as the first-pass lookup before falling
back to Google Translate.

### `review/round_1.xlsx` and `review/round_2.xlsx`
Two-annotator evaluation samples used in `notebook/annotations.ipynb`
to compute inter-annotator reliability and model-vs-consensus F1. Each
file covers 300 nominalised reasons sampled from
`classified_micro_reasons.csv`; round 1 uses the initial classification
prompt, round 2 uses the refined prompt. Place them under
`data/review/` after download.

| Column | Type | Description |
| --- | --- | --- |
| `text` | str | Nominalised reason (same paraphrased form as in `classified_micro_reasons.csv`). |
| `annotator_1` | int 1–10 | Label assigned by the first human annotator. |
| `annotator_2` | int 1–10 | Label assigned by the second human annotator. |
| `consensus` | int 1–10 (nullable) | Consensus label after adjudication. NA for a small number of round-2 rows where reviewers did not converge; these rows are excluded from model-vs-consensus metrics. |
| `Modelo` | int 1–10 (nullable) | Label produced by the LLM for this reason (the model run evaluated in that round). |

## What is NOT hosted

- **Raw open-ended responses** (column `Abierta` in the source Excel).
  Covered by IRB consent; available on reasonable request.
- **Intermediate sentence embeddings** (`processed_data_embeddings.csv`
  in the local working copy). Regeneratable from the raw text and
  embedding model; shipping them would require shipping the respondent
  text.

## Reproducing the derived artifacts from scratch

If you have access to the raw source Excel file, the full chain is:

```bash
python -m cam_peru.extract_arguments \
    --input /path/to/TradyAlt\ Fase1.xlsx \
    --output data/extracted_reasons.csv

python -m cam_peru.run_classification_micro \
    --input data/arguments.csv \
    --output data/classified_micro_reasons.csv

# Then rebuild document_category_labels and composition matrices via
# the notebooks in notebook/ or directly with pandas.
```

Seeds and temperatures are pinned in `cam_peru/config.py`, but
minor stochastic drift in LLM outputs is possible between runs — see
the F1 / inter-rater agreement analysis in `notebook/annotations.ipynb`
for the magnitude observed in practice.
