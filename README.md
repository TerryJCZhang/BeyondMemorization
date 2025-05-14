# Math Benchmark

A comprehensive pipeline for extracting, processing, and evaluating mathematical content from arXiv papers.


## Motivation

- We aim to create a benchmark that focuses on real-world math problems—those that could appear in research papers or on platforms like Stack Overflow, where the goal is to solve practical problems. In contrast, we do not target the highly challenging math problems found in competitions such as the IMO or AMC.

- Our benchmark is designed to be easily extensible and adaptable. As new models are released—some potentially trained on our dataset—we can seamlessly evaluate them using the most up-to-date data, ensuring robustness while avoiding dataset contamination.

- With existing benchmarks becoming saturated, there is a growing need for higher-quality benchmarks that focus on realistic problems. For instance, in coding benchmarks, the trend is shifting from competition-style challenges (e.g., LeetCode, Codeforces) to more practical, real-world coding tasks, such as those faced by coding agents in real applications.


## Overview

This project implements an end-to-end pipeline that:
1. Retrieves papers related to mathematical problems from arXiv, e.g., CS, Math, etc.
2. Extracts and processes LaTeX source code
3. Extracts theorems from the papers
4. Generates question-answer pairs from the theorems
5. Evaluate the capabilities of LLMs on solving the question-answer pairs



## Requirements

- Python 3.12
- Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

- As this repo doesn't require any GPU, it's easy to run the server locally. But make sure you have a    <span style="color:red">latex installation</span>  on your machine, because we ensure the QA pairs in latex format can be directly rendered when we mannually check the theorems.

## Quick Start

Complete pipeline execution, papers from CS as an example:

```bash
#!/bin/bash
OUTPUT_PATH=MATH_2024_12

# Execute the first 3 commands in order
# 1. Retrieve math papers
python helpers/arxiv_retriever.py --year 2024 --month 11 --output $OUTPUT_PATH/papers --max-results 500 --category math

# 2. Extract LaTeX source
python helpers/extract_latex_text.py --input $OUTPUT_PATH/papers --output $OUTPUT_PATH/latex


# 3. Extract theorems
python helpers/extract_theorems.py --input $OUTPUT_PATH/latex --sample_papers 100 --output $OUTPUT_PATH/theorems 

# 4. Generate QA pairs
python helpers/extract_qa.py --input $OUTPUT_PATH/theorems_filtered --sample_papers 20 --output $OUTPUT_PATH/qa_pairs 


# 4. Evaluate the QA pairs
python eval_math.py --model gpt-4o --sample 500 --output $OUTPUT_PATH/results  --verbose --dataset $OUTPUT_PATH/qa_pairs_filtered


# Wait for both parallel processes to complete
wait

echo "Done!"
```
Note: 

- if we set `--no_context`, the model will not use the paper content to answer the question. It's seems surprising that this can work for some questions. We may need to check if the questions are too easy or the model is just amazingly good.

# How to visualize

This step is optional. It's a bit painful to correctly visualize all the text in a latex format.

After we have the dataset and experimental results, we can visualize them by running the following command:

1. Visualize a theorems-dataset locally
```
python visualize.py --mode dataset --path MATH_2024_12/theorems
```

2. or compile it into a pdf
```
python vis_pdf.py --data_path MATH_2024_12/theorems  --mode theorem
```



##  Analyze the results

At default, when evaluating the model, we will save the results in `.jsonl` format.

To analyze the results, you can use the following command:
```
python analysis/model_performance.py --input CS/ --output analysis/plots
```



# TODO

- Make sure that the QA pairs in the arXiv dataset are of high quality, for math papers, it seems to be good. We should check the quality of the QA pairs in other categories, e.g., CS.
- We can also look at math questions from StackOverflow.