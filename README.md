# Time Will Tell: A Longitudinal Analysis on Reasoning-Driven Transformation Against Benchmark Contamination

<div align="center">

#### [📄 Paper](https://arxiv.org/)  |  [🤗 Data (Hugging Face)](https://huggingface.co/)
</div>

## Table of contents

- Overview
- 1) Main experiment — month-by-month QA synthesis and evaluation
- 2) Validation experiment 1 — CLOZE using abstracts
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

This repo includes `pyproject.toml` and `uv.lock`. If you use `uv`, restore from the lockfile. Otherwise create a
virtualenv and install `requirements.txt`.

---


