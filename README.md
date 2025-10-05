# Time Will Tell: A Longitudinal Analysis on Reasoning-Driven Transformation Against Benchmark Contamination

<div align="center">

#### [📄 Paper](arxiv link)  |  [🤗 Data](In Repo) 
</div>

## Overview

This Repo implements an end-to-end pipeline that:
1. Retrieves papers related to scientific problems from arXiv, e.g., CS, Physics, etc.
2. Extracts and processes LaTeX source code
3. Extracts theorems from these papers
4. Generates question-answer pairs from theorems with fixed answers
5. Evaluate the capabilities of LLMs on solving the question-answer pairs

## Requirements

- Python 3.12
- Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

- As this repo doesn't require any GPU, it's easy to run the server locally. But make sure you have a    <span style="color:red">latex installation</span>  on your machine, because we ensure the QA pairs in latex format can be directly rendered when we mannually check the theorems.

## Quick Start

Note that our benchmark is fully automated and refreshable. For example, we can simply run the following script to retrieve the latest papers from May 2024 to December 2024 and evaluate frontier models on them.

```bash
#!/bin/bash
# BeyondMemorization

A research codebase used to retrieve arXiv papers, extract LaTeX and theorems, synthesize
high-quality question–answer (QA) pairs from theorems, and evaluate language models on
those QA tasks.

This repository contains the pipeline and utilities for the project "Beyond Memorization: Reasoning-Driven
Synthesis as a Mitigation Strategy Against Benchmark Contamination". Add paper / dataset links here.

## Highlights

- Month-by-month pipeline that fetches arXiv papers, extracts LaTeX, finds theorems, and synthesizes QA pairs.
- QA generation enforces strict rules (LaTeX formatting, unique answers) to produce renderable, verifiable items.
- Evaluation tools to compare LLMs (OpenAI / OpenRouter / Anthropic) on the generated QA benchmark.

## Repository layout

Top-level files and folders you'll use most often:

- `monthly_qa_pipeline.py` — orchestrates month-wise retrieval → extraction → theorem extraction → QA generation.
- `helpers/` — pipeline steps and prompt definitions:
  - `arxiv_retriever.py` — fetch papers from arXiv
  - `extract_latex_text.py` — extract and normalize LaTeX source
  - `extract_theorems.py` — extract theorems and contextual information
  - `generate_qa.py` — LLM-driven QA synthesis from theorems
  - `prompts.py` — system/user prompts used by the LLMs
- `eval.py` — evaluate LLMs on the QA dataset and produce metrics / logs
- `datacollate.py` — collate per-month outputs into a structured `data/` tree
- `count_qa_pairs.py` — utility to count and summarize QA pairs across outputs
- `requirements.txt` — Python dependencies for the project

Auxiliary folders in this repo include evaluation logs and archived results in
`MainResults-EvalLogMonthlyQA/` and experimental notebooks in `ValidationExp*`.

## Quick start (development / evaluation)

Prerequisites

- Python 3.10+ (code comments reference 3.12; either should be fine). Use a virtual environment.
- A LaTeX engine (e.g. TeX Live / MikTeX) installed locally if you plan to render LaTeX examples.
- API keys for model providers you plan to use (OpenRouter / OpenAI / Anthropic).

Install dependencies

PowerShell (Windows):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Basic pipeline example

1) Generate month-wise QA pairs (creates `output/<category>/<topic>/YYYY/MM/` folders):

```powershell
python monthly_qa_pipeline.py --year 2024 --start 5 --end 12 --categories math,cs --papers-step 100
```

2) Collate produced QA pairs into a central `data/` directory:

```powershell
python datacollate.py --input output --output data
```

3) Count QA pairs and inspect token sizes:

```powershell
python count_qa_pairs.py --input output --json_out qa_count.json --show_exceed_tokens True
```

4) Run evaluation on a dataset (example):

```powershell
python eval.py --dataset data --output results --model o4-mini
```

Note: read each script's top docstring / --help for exact flags.

## Environment variables / API keys

Set required API keys before running LLM-backed steps. The code uses `python-dotenv` to load a `.env` file
if present.

- `OPENROUTER_API_KEY` — commonly used as the OpenRouter API key (also used as an OpenAI backend in code)
- `OPENAI_API_KEY` — optional, direct OpenAI SDK
- `ANTHROPIC_API_KEY` — optional, for Claude models

Example `.env` (DO NOT commit this file):

```text
OPENROUTER_API_KEY=sk-...
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=claude-...
```

Security note: never commit secrets to git. Use a credentials manager for production.

## Data layout and conventions

- `output/<main_cat>/<subcat or topic>/<YYYY>/<MM>/papers` — saved dataset of fetched papers
- `output/.../<YYYY>/<MM>/qa_pairs` — HF dataset of generated QA pairs for that month/topic
- `data/<main_cat>/<YYYY>/<MM>/qa_pairs` — collated dataset produced by `datacollate.py`

Helpers like `count_qa_pairs.py` and `datacollate.py` assume this folder layout.

## Important implementation notes

- Theorem extraction (`helpers/extract_theorems.py`) uses heuristics and regex to find theorem-like environments
  and attempts to preserve context and numbering. It also contains logic to standardize LaTeX so examples
  can be rendered.

- QA generation (`helpers/generate_qa.py`) uses an LLM client and strict prompts (in `helpers/prompts.py`) to
  require LaTeX-formatted question and unique-answer outputs. The generator supports synchronous and
  asynchronous clients and contains retry logic.

- Evaluation (`eval.py`) supports multiple backends (OpenAI/OpenRouter/Anthropic) and provides both
  synchronous and asynchronous querying utilities.

## Troubleshooting & tips

- If a model call fails, confirm your API key environment variables and that the corresponding SDK is installed.
- Logs: pipelines and per-month runs append logs to the `output/...` folder (e.g. `qa_generate.log`,
  `monthly_qa_pipeline.log`) — inspect them for detailed failures.
- Prompt tuning: the QA generation is prompt-sensitive. If you modify prompts in `helpers/prompts.py`, run a
  small sample before re-generating a large batch.

## Reproducibility / tests

- The repository uses deterministic seeds (e.g. `seed=42`) in several places for repeatability of shuffling.
- For a minimal smoke test, run `count_qa_pairs.py` on an `output/` snapshot.

## Citations

If you use the dataset or code, please cite the associated paper (add arXiv / bibtex reference here).

## Optional follow-ups I can implement

- Add a minimal `examples/` folder with a tiny example dataset and a smoke test that runs the pipeline on
  the example (safe, no external API calls).
- Expand the `helpers/` README with per-script examples and typical runtime notes.

If you want one of these additions, tell me which and I will create it.





