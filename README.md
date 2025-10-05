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
- Archive: MainResults-EvalLogMonthlyQA

---

## Overview

This repository contains three separate parts. Each part has the scripts, data and notebooks used for the
corresponding experiments.

- Main experiment: fetch arXiv papers, extract LaTeX/theorems, synthesize LaTeX-formatted question/answer pairs,
  and evaluate models on the generated QA pairs.
- Validation 1 (CLOZE): create fill-in-the-blank questions from paper abstracts and evaluate model predictions.
- Validation 2 (Perturbed LiveCodeBench): apply small, controlled transformations to code problems and measure
  how model outputs change.

The sections below list the main scripts, where outputs are stored, and a few example commands.

---

## 1) Main experiment — month-by-month QA synthesis and evaluation

Summary

This pipeline downloads papers by arXiv category and month, extracts LaTeX, detects theorem statements, and
creates question–answer pairs. QA pairs are stored as datasets and can be evaluated with the included
evaluation harness.

Main scripts and helpers

- `monthly_qa_pipeline.py` — run the end-to-end monthly pipeline.
- `helpers/arxiv_retriever.py` — download paper metadata and source.
- `helpers/extract_latex_text.py` — extract and normalize LaTeX source.
- `helpers/extract_theorems.py` — locate theorem-like environments and extract content.
- `helpers/generate_qa.py` — generate LaTeX-formatted QA pairs (prompts live in `helpers/prompts.py`).
- `datacollate.py` — collect per-month QA datasets into `data/`.
- `count_qa_pairs.py` — report counts and token-size issues.
- `eval.py` — evaluation harness for comparing model outputs.

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

- `ValidationExp1-CLOZEusingAbstracts/generate_and_evaluate_cloze_abstracts.py` — main script.
- `ValidationExp1-CLOZEusingAbstracts/CLOZEonRealMathPapers/` — sample data and result files.

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

- `ValidationExp2-PerturbedLiveCodeBench/livecodebench.py` — run perturbations and evaluations.
- `ValidationExp2-PerturbedLiveCodeBench/analysis.py` — aggregate results and compute metrics.
- `ValidationExp2-PerturbedLiveCodeBench/plotting.py` — generate figures.
- Notebooks: `livecodebench-analysis.ipynb`, `livecodebench-final.ipynb`.

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

## Archive: MainResults-EvalLogMonthlyQA

This folder holds archived evaluation outputs mentioned in the manuscript:

- `eval_deepseek.tar.gz` — DeepSeek-style outputs and logs
- `eval_gemini_and_openai.tar.gz` — Gemini + OpenAI combined outputs
- `eval_llama.tar.gz` — LLaMA-family outputs and logs

Open these archives to inspect raw model outputs and run logs.


