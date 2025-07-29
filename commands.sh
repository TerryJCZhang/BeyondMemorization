#!/bin/bash
OUTPUT_PATH=results

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