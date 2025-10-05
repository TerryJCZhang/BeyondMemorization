# Time Will Tell: A Longitudinal Analysis on Reasoning-Driven Transformation Against Benchmark Contamination

<div align="center">

#### [📄 Paper](https://arxiv.org/)  |  [🤗 Data (Hugging Face)](https://huggingface.co/)
</div>

## Table of contents

- [Overview](#overview)
- [Main experiment — month-wise QA synthesis and evaluation](#1-main-experiment-—-month-wise-qa-synthesis-and-evaluation)
- [Validation experiment 1 — CLOZE using abstracts](#2-validation-experiment-1-—-cloze-using-abstracts)
- [Validation experiment 2 — Perturbed LiveCodeBench](#3-validation-experiment-2-—-perturbed-livecodebench)
- [Environment variables and secrets](#environment-variables-and-secrets)
- [Reproducibility and package management](#reproducibility-and-package-management)
- [Contact and citation](#contact-and-citation)

**Overview:** This repository is organized into three self-contained parts. Each part is independently runnable and
contains scripts, datasets, and notebooks needed to reproduce the associated analyses. The three parts are:

1. Main experiment — month-wise QA synthesis and evaluation: a pipeline that retrieves arXiv papers, extracts
  LaTeX/theorems, synthesizes LaTeX-formatted question–answer pairs, and evaluates language models on those items.
2. Validation experiment 1 — CLOZE using abstracts: a focused validation that generates cloze-style questions from
  paper abstracts and evaluates model performance on fill-in-the-blank tasks.
3. Validation experiment 2 — Perturbed LiveCodeBench: a controlled validation which applies perturbations to code
  problems and measures model robustness to those transformations.

Below is a concise, reviewer-focused README organized around those three parts, with runnable examples and the
locations of the most important scripts and outputs.

Prerequisites (short)

- Python 3.10+ (the code references 3.12 in places)
- A LaTeX engine (TeX Live, MikTeX) if you want to render LaTeX outputs
- API keys for model-backed steps (see "Environment variables" below)

Install dependencies (PowerShell example):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

-------------------------------------------------------------------------------

## 1) Main experiment — month-wise QA synthesis and evaluation

Purpose: retrieve arXiv papers, extract LaTeX/theorems, synthesize QA pairs from theorems, and evaluate LLMs.

Top-level orchestration and important files

- `monthly_qa_pipeline.py` — main driver. For each (category, month) it:
  1. downloads papers with `helpers/arxiv_retriever.py`
  2. extracts LaTeX with `helpers/extract_latex_text.py`
  3. finds theorems using `helpers/extract_theorems.py`
  4. generates QA pairs via `helpers/generate_qa.py`

- `datacollate.py` — collates `output/.../qa_pairs` into a central `data/` directory.
- `count_qa_pairs.py` — counts QA pairs across `output/` and reports token-size issues.
- `eval.py` — evaluation harness that supports OpenAI / OpenRouter backends.

Folder layout (for a single topic/month):

```
output/<main_cat>/<topic>/<YYYY>/<MM>/
  papers/        # fetched papers (datasets saved to disk)
  latex/         # extracted LaTeX files and normalized text
  theorems/      # extracted theorem dataset
  qa_pairs/      # generated QA pairs (HF dataset saved to disk)
  monthly_qa_pipeline.log
  qa_generate.log
```

Quick run (small smoke test — PowerShell):

```powershell
# Run a one-month job for math
python monthly_qa_pipeline.py --year 2024 --start 5 --end 5 --categories math --papers-step 50

# Collate produced QA pairs
python datacollate.py --input output --output data

# Count and inspect
python count_qa_pairs.py --input output --json_out qa_count.json --show_exceed_tokens True

# Evaluate on collated data
python eval.py --dataset data --output results --model o4-mini
```

Logs and troubleshooting

- Per-month logs (e.g., `monthly_qa_pipeline.log`, `qa_generate.log`) live in the corresponding
  `output/.../<YYYY>/<MM>/` folder; inspect them for failures or model API errors.
- Prompts and rules: see `helpers/prompts.py` — QA generation enforces LaTeX formatting and unique answers.
- Use the included `uv.lock` or the provided `requirements.txt` in a venv to pin dependencies.

-------------------------------------------------------------------------------

## 2) Validation experiment 1 — CLOZE using abstracts

Purpose: generate and evaluate cloze-style (fill-in-the-blank) questions derived from paper abstracts.

Location: `ValidationExp1-CLOZEusingAbstracts/`

Key files and contents

- `generate_and_evaluate_cloze_abstracts.py` — main script that builds cloze items from abstracts, optionally
  synthesizes candidate cloze questions, and runs evaluation routines against models.
- `pyproject.toml` and `uv.lock` — local project metadata and lockfile for reproducibility in this experiment folder.
- `CLOZEonRealMathPapers/` — dataset and evaluation artifacts (examples in `realmath_abstracts_with_cloze.json`,
  processed files, and model evaluation JSON outputs such as `evaluation_gpt4omini_results.json`).

How the experiment works (high-level)

1. Load abstracts dataset (the repo includes a processed sample in `CLOZEonRealMathPapers/`).
2. Create cloze prompts by masking a target token/phrase from the abstract.
3. Query models (via `eval.py` or a small wrapper) to predict the masked span.
4. Compute exact-match / edit-distance metrics and save results in the `CLOZEonRealMathPapers/` directory.

Run example (local test):

```powershell
python ValidationExp1-CLOZEusingAbstracts/generate_and_evaluate_cloze_abstracts.py --input ValidationExp1-CLOZEusingAbstracts/CLOZEonRealMathPapers/realmath_abstracts_with_cloze.json --output ValidationExp1-CLOZEusingAbstracts/CLOZEonRealMathPapers/results
```

Outputs and artifacts

- `*_results.json` and `*_scores.json` files in `CLOZEonRealMathPapers/` record per-model outputs and aggregated scores.
- Notebooks in `ValidationExp1-CLOZEusingAbstracts/` demonstrate analysis and reproduction of reported metrics.

This validation is intentionally narrower than the main experiment and uses smaller inputs to make
rapid inspection and manual checking easier for reviewers.

-------------------------------------------------------------------------------

## 3) Validation experiment 2 — Perturbed LiveCodeBench

Purpose: evaluate the effect of controlled perturbations and transformations applied to LiveCodeBench examples,
measuring robustness and generalization of model reasoning on transformed code problems.

Location: `ValidationExp2-PerturbedLiveCodeBench/`

Key files and contents

- `livecodebench.py` — core code for loading LiveCodeBench problems and applying perturbations/transformations.
- `analysis.py` — analysis scripts that compute metrics, aggregate results and produce the tables used in the paper.
- `plotting.py` — utilities to produce the figures used in the manuscript.
- `livecodebench-analysis.ipynb` and `livecodebench-final.ipynb` — interactive notebooks with the full analysis and plots.

How to run (example)

```powershell
# Run a transformation + evaluation pipeline on a small subset
python ValidationExp2-PerturbedLiveCodeBench/livecodebench.py --input ValidationExp2-PerturbedLiveCodeBench/sample_problems.json --output ValidationExp2-PerturbedLiveCodeBench/results --perturbation-type rename_vars

# Run analysis and create summary CSV/plots
python ValidationExp2-PerturbedLiveCodeBench/analysis.py --results ValidationExp2-PerturbedLiveCodeBench/results --out stats.csv
python ValidationExp2-PerturbedLiveCodeBench/plotting.py --stats stats.csv --out plots/
```

Outputs and artifacts

- `results/` — per-run model outputs on transformed problems
- `stats.csv` / `plots/` — aggregated metrics and visualization artifacts
- Notebooks provide reproducible steps to regenerate all figures and tables.

Check script flags for available configuration options.

-------------------------------------------------------------------------------

## Environment variables and secrets

The repository uses `python-dotenv` to load a `.env` file if present. Common variables used by the scripts:

- `OPENROUTER_API_KEY` — OpenRouter / OpenAI backend key used across scripts
- `OPENAI_API_KEY` — optional direct OpenAI SDK key

Example (POSIX):

```bash
export OPENROUTER_API_KEY="your-openrouter-key"
export OPENAI_API_KEY="your-openai-key"
```

PowerShell (Windows):

```powershell
$env:OPENROUTER_API_KEY = "your-openrouter-key"
$env:OPENAI_API_KEY     = "your-openai-key"
```

Security: do not commit keys or `.env` files to the repository. Use a secrets manager for long runs.

-------------------------------------------------------------------------------

## Reproducibility and package management

- The repo includes `pyproject.toml` and an optional `uv.lock` to pin dependencies. We recommend using
  `uv` to restore a reproducible environment if available; otherwise use `pip` in a venv and the provided
  `requirements.txt`.

## Contact and citation

If you use this code or dataset, please cite the associated paper (add arXiv / bibtex here). For questions
about reproducing experiments, open an issue or contact the authors via the paper contact details.

---

If you'd like, I can (A) add runnable example inputs under `examples/` for each experiment so reviewers can
run a fast, local smoke test without API calls, or (B) generate short per-script `--help` usage snippets
collected into a `helpers/README.md`. Which would you prefer?
  

