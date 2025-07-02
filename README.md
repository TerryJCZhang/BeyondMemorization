# PioneerScience: Research-Level Scientific Benchmark from Real-World Research Paper

<div align="center">

#### [📄 Paper](https://arxiv.org/abs/2505.12575)  |  [🤗 RealMath](https://huggingface.co/datasets/ethz-spylab/realmath) 
</div>

This is the official code for [RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics](https://arxiv.org/abs/2505.12575) by Jie Zhang, Cezara Petrui, Kristina Nikolić, Florian Tramèr.



We provide a [HuggingFace dataset](https://huggingface.co/datasets/ethz-spylab/RealMath), which includes 633 samples from *math.arXiv*, 111 samples from *cs.arXiv*, and 542 samples from *Math Stack Exchange*.

![description](teaser.png)


## Motivatation

We aim to address the following question:

**How effective are large language models as assistants for practicing mathematicians today?**

To explore this, our benchmark is designed with the following criteria in mind:

- **Real-world application focus**  
  Content is sourced from authentic mathematical research to reflect real-world use cases.

- **Automated verification**  
  Samples are designed to be easy to verify programmatically for scalable and reliable evaluation.

- **Continuous acquisition**  
  The benchmark is refreshable and designed to avoid dataset contamination over time.







In this work, we present **RealMath**:

- A data pipeline that extracts verifiable mathematical statements from research papers (e.g., arXiv) and mathematical forums (e.g., Stack Exchange), creating a rich corpus of research-level mathematical content.

- An evaluation methodology focused on verifiable answers rather than proof assessment, allowing for automated correctness checking.

- A continually refreshable dataset design, that leverages the vast and growing body of mathematical literature, allowing for regular updates with new content to mitigate contamination concerns.


## Overview

This project implements an end-to-end pipeline that:
1. Retrieves papers related to mathematical problems from arXiv, e.g., CS, Math, etc.
2. Extracts and processes LaTeX source code
3. Extracts theorems from these papers
4. Generates question-answer pairs from theorems with fixed answers
5. Evaluate the capabilities of LLMs on solving the question-answer pairs

![description](pipeline.png)


## Requirements

- Python 3.12
- Dependencies:
  ```bash
  pip install -r requirements.txt
  ```

- As this repo doesn't require any GPU, it's easy to run the server locally. But make sure you have a    <span style="color:red">latex installation</span>  on your machine, because we ensure the QA pairs in latex format can be directly rendered when we mannually check the theorems.

## Quick Start

Note that our benchmark is fully automated and refreshable. For example, we can simply run the following script to retrieve the latest math papers from May 2025 and evaluate frontier models on them.

```bash
#!/bin/bash
OUTPUT_PATH=MATH_2025_5

# 1. Retrieve math papers
python helpers/arxiv_retriever.py --year 2025 --month 5 --output $OUTPUT_PATH/papers --max-results 1000 --category math

# 2. Extract LaTeX source
python helpers/extract_latex_text.py --input $OUTPUT_PATH/papers --output $OUTPUT_PATH/latex

# 3. Extract theorems
python helpers/extract_theorems.py --input $OUTPUT_PATH/latex --output $OUTPUT_PATH/theorems 

# 4. Generate QA pairs
python helpers/generate_qa.py --input $OUTPUT_PATH/theorems --output $OUTPUT_PATH/qa_pairs 

# 5. Evaluate the QA pairs
python eval_math.py --model o4-mini --dataset $OUTPUT_PATH/qa_pairs --output $OUTPUT_PATH/results  &

python eval_math.py --model claude-3.7-sonnet --dataset $OUTPUT_PATH/qa_pairs --output $OUTPUT_PATH/results   &

python eval_math.py --model claude-3.7-sonnet --dataset $OUTPUT_PATH/qa_pairs --use_thinking --parallel 10 --output $OUTPUT_PATH/results &

# Wait for both parallel processes to complete
wait

echo "Done!"
```

# Cite
If you use this code/dataset in your research, please cite the following paper:
```bib
@misc{zhang2025realmathcontinuousbenchmarkevaluating,
      title={RealMath: A Continuous Benchmark for Evaluating Language Models on Research-Level Mathematics}, 
      author={Jie Zhang and Cezara Petrui and Kristina Nikolić and Florian Tramèr},
      year={2025},
      eprint={2505.12575},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.12575}, 
}
```





