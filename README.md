# Zekan

Zekan — a local-first ML trust-audit tool. Measures data-leakage severity, ranks fixes by business cost, and proves repairs. v1: binary classification, tabular, table-first.

## Development

This project uses a dedicated virtual environment to avoid interpreter ambiguity.
Always activate `.venv` before working on zekan — it is the one canonical interpreter.

```bash
# Create (once)
D:\Users\Sakthi\anaconda3\python.exe -m venv .venv

# Activate (every session)
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# or Windows cmd:
.venv\Scripts\activate.bat
# or bash/git-bash:
source .venv/Scripts/activate

# Install (once, or after pulling new deps)
pip install -e ".[dev]"
```

## Install

```bash
pip install -e ".[dev]"
```

## Usage

```bash
zekan audit --data data.csv --config zekan.yaml --model model.pkl
zekan benchmark
zekan report
```
