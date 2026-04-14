#!/usr/bin/env bash
# One-shot reproducibility helper.
#
# Creates a local Python 3.12 virtual environment in .venv, installs every
# dependency pinned in requirements.txt, installs the cam_peru package in
# editable mode, and pulls the spaCy Spanish model + NLTK Spanish stopwords.
#
# After this completes, activate the environment with:
#
#     source .venv/bin/activate
#
# and then either:
#   - run the notebooks under notebook/ (requires the OSF-hosted CSVs in data/)
#   - invoke the CLI entry points (`python -m cam_peru.extract_arguments`,
#     `python -m cam_peru.run_classification`,
#     `python -m cam_peru.run_classification_micro`) — raw Excel required
#
# The raw open-ended responses are NOT redistributed. Download the derived
# CSVs from the OSF project and place them under data/ to reproduce the
# notebooks end-to-end. See data/README.md for the exact file list.
#
# Python version: the repo ships a `.python-version` file pinning 3.12.7.
# If you have pyenv installed, the correct interpreter is selected
# automatically. Otherwise install Python 3.12.x manually (3.13 and 3.14
# are not yet supported by the pinned numpy/blis wheels). Override the
# interpreter explicitly with `PYTHON_BIN=/path/to/python3.12 bash reproduce.sh`.

set -euo pipefail

# ---- Pick an interpreter ----------------------------------------------------
# Priority: explicit PYTHON_BIN env var > pyenv (if installed) > python3 on PATH.
if [ -n "${PYTHON_BIN:-}" ]; then
  :  # user-provided, use as-is
elif command -v pyenv >/dev/null 2>&1; then
  # pyenv reads .python-version automatically; resolve to the absolute path
  # so the subsequent venv is built against the pinned interpreter even if
  # the user's shell doesn't have pyenv's shims at the front of PATH.
  PYTHON_BIN="$(pyenv which python 2>/dev/null || true)"
  if [ -z "$PYTHON_BIN" ]; then
    echo "error: pyenv is installed but cannot resolve a Python interpreter." >&2
    echo "       Run 'pyenv install 3.12.7' and retry." >&2
    exit 1
  fi
else
  PYTHON_BIN="python3"
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "error: $PYTHON_BIN not found on PATH" >&2
  exit 1
fi

py_major_minor=$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')
case "$py_major_minor" in
  3.10|3.11|3.12) ;;
  *)
    echo "error: Python $py_major_minor is not supported." >&2
    echo "       This project requires Python 3.10–3.12 (tested on 3.12.7)." >&2
    echo "       Python 3.13+ breaks the pinned blis/thinc wheels used by spaCy." >&2
    echo "       With pyenv:  pyenv install 3.12.7 && pyenv local 3.12.7" >&2
    echo "       Or pass:     PYTHON_BIN=/path/to/python3.12 bash reproduce.sh" >&2
    exit 1
    ;;
esac

echo ">> Using interpreter: $PYTHON_BIN (Python $py_major_minor)"

if [ ! -d .venv ]; then
  echo ">> Creating virtual environment in .venv/"
  "$PYTHON_BIN" -m venv .venv
fi

# shellcheck source=/dev/null
source .venv/bin/activate

echo ">> Upgrading pip"
python -m pip install --upgrade pip >/dev/null

echo ">> Installing pinned dependencies"
python -m pip install -r requirements.txt

echo ">> Installing cam_peru package (editable)"
python -m pip install -e .

echo ">> Downloading spaCy Spanish model"
python -m spacy download es_core_news_md

echo ">> Downloading NLTK Spanish stopwords"
python -c "import nltk; nltk.download('stopwords', quiet=True)"

echo ">> Running smoke-test imports"
python - <<'PY'
import importlib, sys
mods = [
    "cam_peru.config",
    "cam_peru.llm_client",
    "cam_peru.classification",
    "cam_peru.process_text",
    "cam_peru.embeddings",
    "cam_peru.semantic_map",
    "cam_peru.word_clouds",
]
for m in mods:
    importlib.import_module(m)
    print(f"   ok  {m}")
print("all cam_peru submodules import cleanly.")
PY

cat <<MSG

done. activate the environment in subsequent shells with:

    source .venv/bin/activate

next steps:
  1. download the OSF-hosted CSVs listed in data/README.md into data/
  2. (optional) copy .env.example to .env and fill in Azure credentials
     — only needed to regenerate classifications from raw text
  3. open notebook/results.ipynb to reproduce the figures

MSG
