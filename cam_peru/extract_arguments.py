"""
Reason-extraction pass: condense each free-text response into a list of
nominalised Spanish reasons. The output is consumed by the micro-classification
step (``run_classification_micro``).

Usage
-----
    python -m cam_peru.extract_arguments \\
        --input provided_docs/TradyAlt\\ Fase1.xlsx \\
        --output data/extracted_reasons.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cam_peru.classification import extract_reasons
from cam_peru.llm_client import get_azure_client


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--sheet", default="Completos")
    p.add_argument("--technique-col", default="Técnica")
    p.add_argument("--response-col", default="Abierta")
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_excel(args.input, sheet_name=args.sheet, header=[0, 1])
    df.columns = df.columns.droplevel(1)
    df = df[[args.technique_col, args.response_col]]

    client = get_azure_client()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as f:
        for index, row in df.iterrows():
            therapy = row[args.technique_col]
            reasons = row[args.response_col]
            response = extract_reasons(reasons, therapy=therapy, client=client)
            f.write(f"{index}; {therapy}; {reasons}; {response}\n")
            print(f"[{index}] {therapy}: {response}")


if __name__ == "__main__":
    main()
