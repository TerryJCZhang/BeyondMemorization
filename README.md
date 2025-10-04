# Time Will Tell: A Longitudinal Analysis on Reasoning-Driven Transformation Against Benchmark Contamination

<div align="center">

#### [📄 Paper](arxiv link)  |  [🤗 Data](In Repo) 
</div>

Code for [Beyond Memorization: Reasoning-Driven Synthesis as a Mitigation Strategy Against Benchmark Contamination](arxiv link)

[Dataset](data link), 1643 Theorem+QAs synthesized from 4235 papers from *math.arXiv*, 16042 papers from *physics.arXiv*

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
OUTPUT_PATH=results/

# 1. Process papers and synthesize QAs for Maths, Physics, CS using month-wise QA generation pipeline
python monthly_qa_pipeline.py --year 2024 --start 5 --end 12

# 5. Evaluate the QA pairs
python eval_math.py --model o4-mini --dataset $OUTPUT_PATH --output $OUTPUT_PATH  &

python eval_math.py --model deepseek-r1-0528 --dataset $OUTPUT_PATH --output $OUTPUT_PATH   &

python eval_math.py --model deepseek-r1-0528 --dataset $OUTPUT_PATH --use_thinking --parallel 10 --output $OUTPUT_PATH/results &

# Wait for both parallel processes to complete
wait

echo "Done!"





