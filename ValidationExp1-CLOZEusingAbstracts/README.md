# CLOZE Question Generation & Evaluation

This project generates CLOZE questions (fill-in-the-blank) from academic papers and evaluates LLM performance using multiple metrics: binary scoring (LLM as judge), ROUGE-L, BLEU, exact match, and per-blank F1.

## Quick Start

### 1. Generate CLOZE Questions

Create a custom Python script to define your prompts and source material:

```python
#!/usr/bin/env python3
import json
from openai import AsyncOpenAI
import asyncio

# Configure your API keys
client = AsyncOpenAI(api_key="your-api-key")

# Define your paper sources (with dates for cutoff tracking)
papers = [
    {
        "link": "http://arxiv.org/abs/...",
        "title": "Your Paper Title",
        "date": "2024-01-15",
        # Choose ONE of the following for source material:
        "abstract": "The abstract text here...",  # or use this
        # "full_text": "The full paper text...",  # or use this
        # "section": "Introduction text from specific section...",  # or use this
    }
]

# Define your CLOZE generation prompt (system message)
# This is where you customize HOW blanks are created
system_prompt = """Generate a CLOZE question with exactly 5 blanks [blank1] through [blank5].
Each blank should be:
- A self-contained, meaningful expression (max 5 words)
- Extracted from the provided text
- Representative of key concepts

Return JSON with:
{
    "cloze_question": "text with [blank1]...",
    "answer1": "...",
    "answer2": "...",
    ...
}"""

async def generate_cloze(paper):
    """Generate CLOZE question for a paper"""
    # Choose which text to use
    source_text = paper.get("abstract") or paper.get("full_text") or paper.get("section")
    
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create CLOZE question from:\n{source_text}"}
        ],
        temperature=0.7
    )
    
    return json.loads(response.choices[0].message.content)

# Run generation
async def main():
    results = []
    for paper in papers:
        result = await generate_cloze(paper)
        result.update({"link": paper["link"], "title": paper["title"], "date": paper["date"]})
        results.append(result)
    
    with open("cloze_questions.json", "w") as f:
        json.dump(results, f, indent=2)

asyncio.run(main())
```

### 2. Evaluate Models

```bash
# Run evaluation using the main script
uv run generate_and_evaluate_cloze_abstracts.py
```

This evaluates pre-configured models and computes all metrics.

## Customization Guide

### Choosing Source Material

You can generate CLOZE from:
- **Abstracts** (concise, essential concepts)
- **Full text** (detailed, comprehensive coverage)
- **Specific sections** (Introduction, Methods, Results, etc.)
- **Mixed** (combine multiple sources)

### Customizing Prompt Instructions

Modify the system prompt to control blank difficulty:

```python
# For simpler, more common terms:
"Each blank should be a common technical term (1-3 words)"

# For complex concepts:
"Each blank should be a sophisticated concept or proper noun (max 5 words)"

# For specific domains:
"Blanks should focus on [domain]-specific terminology and concepts"

# For specific properties:
"Blank1: mathematical concept, Blank2: theorem name, Blank3: property..."
```

### Controlling Evaluation Metrics

The evaluation computes:
- **Binary Score**: LLM judges if answer is semantically equivalent (0 or 1)
- **ROUGE-L**: Longest common subsequence overlap
- **BLEU**: N-gram precision (1-gram to 4-gram)
- **Exact Match**: Case-insensitive exact string match
- **Per-Blank F1**: Token-level F1 score (precision × recall of word overlap)

All metrics are computed **per-blank** and **per-question**.

## Data Structure

### Input: Papers JSON
```json
[
  {
    "link": "http://arxiv.org/abs/...",
    "title": "Paper Title",
    "date": "2024-01-15",
    "abstract": "Paper abstract or text...",
    "error": null
  }
]
```

### Output: Results JSON
```json
[
  {
    "link": "...",
    "title": "...",
    "date": "...",
    "before_cutoff": true,
    "cloze_question": "Text with [blank1]...",
    "correct_answers": ["answer1", "answer2", ...],
    "model_answers": ["model_answer1", ...],
    "blank_binary_scores": [0, 1, 1, ...],
    "blank_rouge_scores": [0.5, 0.8, ...],
    "blank_bleu_scores": [0.2, 0.6, ...],
    "blank_exact_match": [0, 1, 0, ...],
    "blank_f1_scores": [0.3, 0.9, 0.2, ...],
    "total_binary_score": 3,
    "total_exact_match": 0,
    "avg_rouge_score": 0.6,
    "avg_bleu_score": 0.4,
    "avg_f1_score": 0.5
  }
]
```

### Output: Scores JSON
```json
{
  "before_cutoff": {
    "questions": 280,
    "binary_points": 474,
    "binary_max_points": 1400,
    "binary_pp": 33.86,
    "rouge_l_pp": 39.10,
    "bleu_pp": 16.43,
    "exact_match_pp": 17.5,
    "f1_pp": 35.66
  },
  "after_cutoff": { ... },
  "overall": { ... }
}
```

## Filtering by Cutoff Date

The system automatically categorizes papers as `before_cutoff` or `after_cutoff` based on model training dates:

- **GPT-4o-mini**: Oct 1, 2023
- **Llama 3.1 405B**: July 1, 2023
- **Claude 3.5 Sonnet**: April 1, 2024
- **Gemini 2 Flash**: Custom (define in config)

Modify these in [generate_and_evaluate_cloze_abstracts.py](generate_and_evaluate_cloze_abstracts.py#L37-L43).

## Workflow Example

1. **Prepare papers**: Create JSON with paper metadata and text source
   ```bash
   python prepare_papers.py --input papers.csv --output papers.json
   ```

2. **Generate CLOZE questions**: Run generation script with custom prompt
   ```bash
   python my_cloze_generator.py
   ```

3. **Evaluate models**: Run the main evaluation
   ```bash
   uv run generate_and_evaluate_cloze_abstracts.py
   ```

4. **Analyze results**: Check `evaluation_*_results.json` and `evaluation_*_scores.json`

## Environment Setup

This project uses UV for Python dependency management:

```bash
# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows

# Install dependencies
uv sync

# Run scripts
uv run script_name.py
```

## API Keys Required

Set environment variables:
```bash
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-..."
```

## Project Structure

```
.
├── generate_and_evaluate_cloze_abstracts.py  # Main evaluation script
├── CLOZEonRealMathPapers/
│   ├── realmath_abstracts_with_cloze.json
│   ├── evaluation_gpt4omini_results.json
│   ├── evaluation_gpt4omini_scores.json
│   └── ...
├── CLOZEonPhysMathArXiv/
│   └── (similar structure)
└── pyproject.toml
```

## License & Citation

If using this in research, please cite appropriately and reference the paper dates for reproducibility and memorization analysis.
