# Single-step Baselines

This repository contains small inference scripts and example result outputs for retrieval/embedding baselines used in experiments. The top-level scripts are:

- `bm25_infer.py`
- `hybrid_infer.py`
- `semantic_infer.py`

Run these scripts to generate retrieval outputs; results are saved in the `results_*` folders.

## Results layout

Results are stored in dataset- and model-specific subfolders and use line-delimited JSON (`.jsonl`):

- `results_2wiki/` — example outputs grouped by model size (`3b`, `7b`) and method (`bm25.jsonl`, `hybrid.jsonl`, `semantic.jsonl`).
- `results_hotpotqa/` — experiment outputs for HotpotQA.
- `results_musique/` — experiment outputs for the Musique dataset.

Each `.jsonl` file contains one JSON object per line representing a query and its retrieved candidates/scores.

## Requirements

- Python 3.8+ (or your preferred 3.x)
- Typical libraries used in retrieval/evaluation workflows (install from your project's `requirements.txt` if available). Example packages that may be required: `tqdm`, `jsonlines`, `numpy`.

Create a virtual environment and install dependencies, for example:

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt  # if you have a requirements file
```

If no `requirements.txt` exists, install minimal tooling:

```bash
pip install tqdm jsonlines
```

## Quick start

1. Inspect a script's CLI and options:

```bash
python bm25_infer.py --help
python hybrid_infer.py --help
python semantic_infer.py --help
```

2. Run an inference script (example):

```bash
python bm25_infer.py [options]
```

3. View results (example):

```bash
# show first 5 lines (Windows PowerShell)
Get-Content results_2wiki\3b\bm25.jsonl -TotalCount 5

# or with Python
python -c "import json;import itertools
print([json.loads(l) for l in itertools.islice(open('results_2wiki/3b/bm25.jsonl'),5)])"
```

