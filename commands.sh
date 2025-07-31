#!/bin/bash
OUTPUT_PATH=results/

export OPENROUTER_API_KEY=your_key

# 1. Retrieve math papers
python arxiv_retriever.py --year 2024 --month 6 --categories cs, math --output output --max-results 100

# 2. Extract LaTeX source
python helpers/extract_latex_text.py --input $OUTPUT_PATH --output $OUTPUT_PATH

# 3. Extract theorems
python helpers/extract_theorems.py --input $OUTPUT_PATH --output $OUTPUT_PATH

# 4. Generate QA pairs
python helpers/generate_qa.py --input $OUTPUT_PATH --output $OUTPUT_PATH

# Or end to end QA pipeline command
python monthly_qa_pipeline.py --year 2024 --start 5 --end 12

# 5. Evaluate the QA pairs
set -euo pipefail

MODELS=(
  "deepseek-r1"
  "deepseek-r1-0528"
  "gemini-2.0-flash"
  "gemini-2.0-pro"
  "o3-mini"
  "o4-mini"
  "llama-4-scout"
  "llama-3.3-70b"
)

PARALLEL=60


LOG_DIR="./logs_eval"
FAILED_LOG="$LOG_DIR/failed_models.txt"
mkdir -p "$LOG_DIR"
rm -f "$FAILED_LOG"


for model in "${MODELS[@]}"; do
  {
    echo "Running $model with --parallel $PARALLEL"
    python eval_math.py --model "$model" --parallel "$PARALLEL" \
      &> "$LOG_DIR/${model//\//_}.log"
    echo "$model completed successfully"
  } || {
    echo "$model failed at $(date)" | tee -a "$FAILED_LOG"
  } &
done

wait

if [[ -s "$FAILED_LOG" ]]; then
  echo "Some models failed. See $FAILED_LOG for details."
else
  echo "All evaluations completed successfully!"
fi