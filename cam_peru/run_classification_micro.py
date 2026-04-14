"""
Micro-classification pass: assign each extracted nominalised reason to the
single most specific category in the 10-category schema.

The output of this script is what powers the downstream composition matrix,
similarity network, and clustering analyses — see the README and
``docs/methodology.md``.

Usage
-----
    python -m cam_peru.run_classification_micro \\
        --input data/arguments.csv \\
        --output data/classified_micro_reasons.csv \\
        --reason-col reason
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cam_peru.classification import (
    MICRO_REASON_PROMPT,
    classify_reason,
)
from cam_peru.llm_client import get_azure_client


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--reason-col",
        default="reason",
        help="Column in the input CSV containing the extracted nominalised reason.",
    )
    p.add_argument(
        "--separator",
        default="|",
        help="Separator for the output CSV (default: '|').",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    client = get_azure_client()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as f:
        for index, row in df.iterrows():
            try:
                reason = row[args.reason_col]
                codes = classify_reason(
                    reason,
                    client=client,
                    prompt_template=MICRO_REASON_PROMPT,
                )
                f.write(
                    f"{row.name}{args.separator}{reason}{args.separator}{codes}\n"
                )
                print(f"[{row.name}] {reason!r} -> {codes}")
            except Exception as exc:  # noqa: BLE001 — we log & keep going
                print(f"[{row.name}] FAILED: {exc}")


if __name__ == "__main__":
    main()
