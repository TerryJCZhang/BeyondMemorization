# Time Will Tell: A Longitudinal Analysis on Reasoning-Driven Transformation Against Benchmark Contamination

<div align="center">

#### [📄 Paper](https://arxiv.org/)  |  [🤗 Data (Hugging Face)](https://huggingface.co/)
</div>

## Table of contents

- Overview
- 1) Main experiment — month-by-month QA synthesis and evaluation of selected models
- 2) Validation experiment 1 — CLOZE QA based on paper abstracts
- 3) Validation experiment 2 — Perturbed LiveCodeBench
- Environment variables
- Package management

---

## Overview

This repository contains three related experiments. Each experiment includes the scripts, datasets and notebooks
used to reproduce the results or to run follow-up analyses.

- Main experiment: download arXiv papers, extract theorem-like content from LaTeX, generate LaTeX-formatted question/answer pairs, and run evaluations on the resulting datasets.
- Validation 1 (CLOZE): produce fill-in-the-blank items from abstracts and measure model predictions.
- Validation 2 (Perturbed LiveCodeBench): apply small code transformations and compare model behavior on original vs transformed problems.

The sections below point to the main scripts, where outputs are written, and a few example commands you can use to get started.

---

## 1) Main experiment — month-by-month QA synthesis and evaluation

Summary

This pipeline downloads papers by arXiv category and month, extracts LaTeX, detects theorem statements, and
creates question–answer pairs. QA pairs are stored as datasets and can be evaluated with the included
evaluation harness.

Main scripts and helpers

- `monthly_qa_pipeline.py` — orchestrates the month/topic pipeline: download papers, extract LaTeX, detect theorem statements, generate QA pairs, and save per-month outputs.
- `helpers/arxiv_retriever.py` — query arXiv and download paper metadata and source archives into the per-month `papers/` folder.
- `helpers/extract_latex_text.py` — extract and normalize LaTeX source from downloaded archives, producing cleaned `.tex` fragments.
- `helpers/extract_theorems.py` — locate theorem-like environments and extract cleaned statements with nearby context for QA generation.
- `helpers/generate_qa.py` — assemble prompts and call the LLM client to produce LaTeX-formatted question/answer pairs from theorems.
- `helpers/prompts.py` — prompt templates and output-format constraints used by the QA generator.
- `datacollate.py` — merge per-month `qa_pairs` datasets into consolidated `data/` folders for evaluation.
- `count_qa_pairs.py` — scan `output/` to count QA pairs, report per-month summaries and token-size warnings.
- `eval.py` — evaluation harness: run model queries on datasets and save per-model outputs and scoring files.

Folder layout (one topic/month)

```
output/<main_cat>/<topic>/<YYYY>/<MM>/
  papers/
  latex/
  theorems/
  qa_pairs/
  monthly_qa_pipeline.log
  qa_generate.log
```

Example commands (PowerShell)

```powershell
# Run pipeline for May 2024 (math)
python monthly_qa_pipeline.py --year 2024 --start 5 --end 5 --categories math --papers-step 50

# Collate outputs into data/
python datacollate.py --input output --output data

# Count QA pairs and check token sizes
python count_qa_pairs.py --input output --json_out qa_count.json --show_exceed_tokens True

# Evaluate models on the collated dataset
python eval.py --dataset data --output results --model o4-mini
```

If you hit problems, check the per-month logs under `output/`. Prompt templates are in `helpers/prompts.py`.

---

## 2) Validation experiment 1 — CLOZE using abstracts

Summary

Create cloze (fill-in-the-blank) items from paper abstracts and evaluate how models predict the missing text.

Where to look

- `ValidationExp1-CLOZEusingAbstracts/generate_and_evaluate_cloze_abstracts.py` — build cloze items from abstracts and run evaluations across models.
- `ValidationExp1-CLOZEusingAbstracts/CLOZEonRealMathPapers/realmath_abstracts_with_cloze.json` — example input dataset used in validation runs.
- Notebooks in the folder — exploratory analysis and visualization of results.

Example

```powershell
python ValidationExp1-CLOZEusingAbstracts/generate_and_evaluate_cloze_abstracts.py --input ValidationExp1-CLOZEusingAbstracts/CLOZEonRealMathPapers/realmath_abstracts_with_cloze.json --output ValidationExp1-CLOZEusingAbstracts/CLOZEonRealMathPapers/results
```

Outputs

- `*_results.json`, `*_scores.json` — per-model outputs and metrics.
- Example notebooks for analysis are included in the folder.

---

## 3) Validation experiment 2 — Perturbed LiveCodeBench

Summary

Apply controlled transformations (e.g., rename variables, reformatting) to LiveCodeBench problems and compare
model behavior on original vs transformed problems.

Where to look

- `ValidationExp2-PerturbedLiveCodeBench/livecodebench.py` — generate perturbations and run model evaluations on the problems.
- `ValidationExp2-PerturbedLiveCodeBench/analysis.py` — aggregate evaluation outputs and compute summary metrics.
- `ValidationExp2-PerturbedLiveCodeBench/plotting.py` — create visualizations from aggregated stats.
- `ValidationExp2-PerturbedLiveCodeBench/livecodebench-analysis.ipynb` — interactive analysis notebook.
- `ValidationExp2-PerturbedLiveCodeBench/livecodebench-final.ipynb` — finalized analysis and figures.

Example

```powershell
# Apply a rename-vars perturbation and run evaluation
python ValidationExp2-PerturbedLiveCodeBench/livecodebench.py --input ValidationExp2-PerturbedLiveCodeBench/sample_problems.json --output ValidationExp2-PerturbedLiveCodeBench/results --perturbation-type rename_vars

# Aggregate results and plot
python ValidationExp2-PerturbedLiveCodeBench/analysis.py --results ValidationExp2-PerturbedLiveCodeBench/results --out stats.csv
python ValidationExp2-PerturbedLiveCodeBench/plotting.py --stats stats.csv --out plots/
```

Outputs

- `results/` — saved model outputs for transformed problems
- `stats.csv` / `plots/` — aggregated metrics and visualizations

---

## Reproducibility — quick guide

If you want to run the experiments or inspect outputs quickly, the steps below reproduce the main pipeline and evaluation used for the paper.

### What we share
- A small subset of the generated QA dataset is included with the submission (six QA pairs per month). The full generated set will be published on Hugging Face after review. The released subset (CC-BY 4.0) contains 1,643 QA pairs (856 math, 787 physics). No personal data or offensive content is included.
- The supplementary materials include the code, prompt templates, and evaluation logs used for the reported runs.

### Quick reproduction steps
1. Create an environment and install dependencies:

```bash
# POSIX
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Provide API credentials (if running model-backed steps). The code reads a local `.env` file or environment variables. Common variables:

```text
OPENROUTER_API_KEY  # OpenRouter / OpenAI backend key
OPENAI_API_KEY      # Optional: direct OpenAI SDK key
```

3. Run a short, month-local pipeline (smoke test):

```powershell
python monthly_qa_pipeline.py --year 2024 --start 5 --end 5 --categories math --papers-step 50
```

Check the month folder (e.g. `output/math/.../2024/05/`) for `papers/`, `theorems/`, and `qa_pairs/`.

4. Collate and summarize outputs:

```powershell
python datacollate.py --input output --output data
python count_qa_pairs.py --input output --json_out qa_count.json
```

5. Run evaluation on the collated dataset:

```powershell
python eval.py --dataset data --output results --model o4-mini
```
---

## Environment variables

The code reads a local `.env` file if present. The scripts commonly use:

- `OPENROUTER_API_KEY` — OpenRouter / OpenAI backend key
- `OPENAI_API_KEY` — optional, direct OpenAI SDK key

POSIX example:

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
export OPENAI_API_KEY="your-openai-key"
```

PowerShell example:

```powershell
$env:OPENROUTER_API_KEY = "your-openrouter-key"
$env:OPENAI_API_KEY     = "your-openai-key"
```

Keep keys out of source control.

---

## Package management

This repository uses a unified environment configuration managed through `pyproject.toml` and `uv.lock` at the root level.

### Setup with uv (recommended)

```powershell
# Install dependencies
uv sync

# Activate the virtual environment
.venv\Scripts\activate
```

### Setup with pip

```powershell
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -e .
```

### Development dependencies

```powershell
# Install with dev dependencies
uv sync --all-extras

# Or with pip
pip install -e ".[dev]"
```

All subprojects (MainExp, ValidationExp1, ValidationExp2) use the same unified environment from the root `pyproject.toml`.

---


