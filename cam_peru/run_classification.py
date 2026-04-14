"""
Document-level LLM classification of open-ended CAM justifications.

For each (technique, open-ended response) pair in the sampled subset this
script calls Azure OpenAI with :data:`cam_peru.classification.DOCUMENT_LEVEL_PROMPT`
and writes a ``doc_id ; technique ; response ; [category_codes]`` row to the
output CSV.

Usage
-----
    python -m cam_peru.run_classification \\
        --input provided_docs/TradyAlt\\ Fase1.xlsx \\
        --output data/classified_reasons.csv \\
        --sample 300 --random-state 42

Credentials are read from the environment — see ``.env.example``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cam_peru.classification import (
    DOCUMENT_LEVEL_PROMPT,
    classify_reason,
)
from cam_peru.config import (
    CLASSIFICATION_RANDOM_STATE,
    CLASSIFICATION_SAMPLE_SIZE,
)
from cam_peru.llm_client import get_azure_client


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to the source Excel file (multi-level header expected).",
    )
    p.add_argument(
        "--sheet",
        default="Completos",
        help="Sheet name inside the source Excel file (default: Completos).",
    )
    p.add_argument(
        "--technique-col",
        default="Técnica",
        help="Column name for the technique label (default: Técnica).",
    )
    p.add_argument(
        "--response-col",
        default="Abierta",
        help="Column name for the open-ended response (default: Abierta).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the CSV where classifications will be appended.",
    )
    p.add_argument("--sample", type=int, default=CLASSIFICATION_SAMPLE_SIZE)
    p.add_argument("--random-state", type=int, default=CLASSIFICATION_RANDOM_STATE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_excel(args.input, sheet_name=args.sheet, header=[0, 1])
    # Drop the second header row — only the top level is meaningful.
    df.columns = df.columns.droplevel(1)
    df = df[[args.technique_col, args.response_col]]
    if args.sample:
        df = df.sample(n=args.sample, random_state=args.random_state)

    client = get_azure_client()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as f:
        for index, row in df.iterrows():
            therapy = row[args.technique_col]
            reasons = row[args.response_col]
            codes = classify_reason(
                reasons,
                therapy=therapy,
                client=client,
                prompt_template=DOCUMENT_LEVEL_PROMPT,
            )
            f.write(f"{index}; {therapy}; {reasons}; {codes}\n")
            print(f"[{index}] {therapy} -> {codes}")


if __name__ == "__main__":
    main()
