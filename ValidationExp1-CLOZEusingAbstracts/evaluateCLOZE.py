#!/usr/bin/env python3
"""
Evaluate CLOZE questions with comprehensive metrics:
- Semantic equivalence (binary scoring via ROUGE-L + BLEU thresholds)
- ROUGE-L scores (continuous metric)
- BLEU scores (continuous metric)
- Exact Match (full answer and per-blank)
- Token-level F1 scores
"""

import asyncio
import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.0.0",
#     "aiohttp",
#     "rouge-score>=0.1.2",
#     "nltk>=3.8",
#     "tqdm>=4.65.0",
# ]
# ///

from openai import AsyncOpenAI
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Configuration
CUTOFF_DATE_OPENAI = datetime(2023, 10, 1)
CUTOFF_DATE_LLAMA = datetime(2023, 7, 1)
CUTOFF_DATE_CLAUDE = datetime(2024, 4, 1)

DEFAULT_MODEL_CONCURRENCY = 50
DEFAULT_JUDGE_CONCURRENCY = 100
DEFAULT_API_TIMEOUT = 120  # seconds

# Default model configurations
DEFAULT_MODELS = [
    {
        "name": "gpt-4o-mini",
        "display_name": "GPT-4o-mini",
        "api": "openai",
        "cutoff_date": CUTOFF_DATE_OPENAI,
        "output_prefix": "gpt4omini"
    },
    {
        "name": "meta-llama/llama-3.1-405b-instruct",
        "display_name": "Llama-3.1-405B",
        "api": "openrouter",
        "cutoff_date": CUTOFF_DATE_LLAMA,
        "output_prefix": "llama31_405b"
    },
    {
        "name": "anthropic/claude-3.5-sonnet",
        "display_name": "Claude-3.5-Sonnet",
        "api": "openrouter",
        "cutoff_date": CUTOFF_DATE_CLAUDE,
        "output_prefix": "claude35sonnet"
    }
]


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase and strip whitespace"""
    return text.lower().strip()


def tokenize(text: str) -> set:
    """Simple tokenization: split on whitespace, lowercase"""
    return set(normalize_text(text).split())


def exact_match(reference: str, hypothesis: str) -> int:
    """
    Calculate exact match score (1 if exact match, 0 otherwise)
    Case-insensitive comparison after stripping whitespace
    """
    return 1 if normalize_text(reference) == normalize_text(hypothesis) else 0


def f1_score(reference: str, hypothesis: str) -> float:
    """
    Calculate token-level F1 score between reference and hypothesis
    
    F1 = 2 * (precision * recall) / (precision + recall)
    where:
    - precision = |reference ∩ hypothesis| / |hypothesis|
    - recall = |reference ∩ hypothesis| / |reference|
    """
    ref_tokens = tokenize(reference)
    hyp_tokens = tokenize(hypothesis)
    
    if len(ref_tokens) == 0 and len(hyp_tokens) == 0:
        return 1.0  # Both empty, perfect match
    
    if len(ref_tokens) == 0 or len(hyp_tokens) == 0:
        return 0.0  # One empty, one not
    
    common = ref_tokens & hyp_tokens
    
    if len(common) == 0:
        return 0.0
    
    precision = len(common) / len(hyp_tokens)
    recall = len(common) / len(ref_tokens)
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_rouge_l(reference: str, hypothesis: str) -> float:
    """
    Calculate ROUGE-L score between reference and hypothesis
    ROUGE-L measures longest common subsequence, good for semantic similarity
    Returns F1 score (0-1)
    """
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores['rougeL'].fmeasure


def calculate_bleu(reference: str, hypothesis: str) -> float:
    """
    Calculate BLEU score between reference and hypothesis
    BLEU measures n-gram precision, good for exact matching
    Returns score (0-1) with smoothing for short texts
    """
    # Tokenize
    reference_tokens = reference.lower().split()
    hypothesis_tokens = hypothesis.lower().split()

    # Use smoothing function for short texts
    smoothing = SmoothingFunction().method1

    # Calculate BLEU with weights for 1-gram to 4-gram
    weights = (0.5, 0.3, 0.15, 0.05)

    try:
        score = sentence_bleu([reference_tokens], hypothesis_tokens,
                              weights=weights, smoothing_function=smoothing)
        return score
    except:
        return 0.0


def compute_all_metrics_for_blank(reference: str, hypothesis: str) -> Dict:
    """
    Compute all metrics for a single blank
    Returns dict with exact_match, f1, rouge_l, and bleu scores
    """
    em = exact_match(reference, hypothesis)
    f1 = f1_score(reference, hypothesis)
    rouge = calculate_rouge_l(reference, hypothesis)
    bleu = calculate_bleu(reference, hypothesis)
    
    # Binary score based on thresholds
    ROUGE_L_MIN = 0.80
    BLEU_MIN = 0.50
    binary_correct = bool(em or (rouge >= ROUGE_L_MIN and bleu >= BLEU_MIN))
    
    return {
        'exact_match': em,
        'f1_score': f1,
        'rouge_l': rouge,
        'bleu': bleu,
        'binary_correct': 1 if binary_correct else 0
    }


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime"""
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d")
    except:
        return datetime(1900, 1, 1)


async def ask_model(
        client: AsyncOpenAI,
        title: str,
        link: str,
        cloze_question: str,
        num_blanks: int,
        model_name: str,
        api_type: str,
        semaphore: asyncio.Semaphore,
        timeout: int
) -> List[str]:
    """Ask model to recall paper and fill blanks"""
    async with semaphore:
        try:
            # Build answer template
            answer_template = "\n".join([f"blank{i}: <answer>" for i in range(1, num_blanks + 1)])
            
            prompt = (
                f"Title: {title}\n"
                f"Link: {link}\n\n"
                f"First, try to recall this paper from your training data based on the title.\n"
                f"Then, fill in the blanks to the best of your memory.\n\n"
                f"IMPORTANT:\n"
                f"- Answer from MEMORY of what you learned during training\n"
                f"- Each answer: 1-6 words maximum\n"
                f"- If you don't remember, make your best educated guess based on the context\n\n"
                f"Text with blanks:\n{cloze_question}\n\n"
                f"Provide your answers:\n{answer_template}"
            )

            response = await asyncio.wait_for(
                client.responses.create(
                    model=model_name,
                    input=[
                        {"role": "system", "content": [{"type": "input_text", "text": "You are an expert researcher deeply familiar with arXiv papers. Each question you are asked, including any QA or CLOZE task, comes from an existing arXiv paper. Follow instructions carefully and answer concisely."}]},
                        {"role": "user", "content": [{"type": "input_text", "text": prompt}]}
                    ]
                ),
                timeout=timeout
            )

            # Extract text
            content = getattr(response, "output_text", None) or ""
            if not content and hasattr(response, "output") and response.output:
                for item_output in response.output:
                    if hasattr(item_output, "content") and item_output.content:
                        for part in item_output.content:
                            if hasattr(part, "text") and part.text:
                                content += part.text

            content = (content or "").strip()

            # Parse answers
            answers = []
            for i in range(1, num_blanks + 1):
                marker = f"blank{i}:"
                lower = content.lower()
                if marker in lower:
                    start = lower.index(marker) + len(marker)
                    rest = content[start:].strip()
                    end = len(rest)
                    for j in range(i + 1, num_blanks + 1):
                        next_marker = f"blank{j}:"
                        idx = rest.lower().find(next_marker)
                        if idx != -1:
                            end = min(end, idx)
                    answer = rest[:end].strip().split('\n')[0].strip()
                    answers.append(answer)
                else:
                    answers.append("NO_ANSWER")

            return answers if len(answers) == num_blanks else ["NO_ANSWER"] * num_blanks

        except asyncio.TimeoutError:
            tqdm.write(f"  ⏱️  Model timeout for: {title[:50]}")
            return ["TIMEOUT"] * num_blanks
        except Exception as e:
            tqdm.write(f"  ❌ Model error: {str(e)[:60]}")
            return ["ERROR"] * num_blanks


async def evaluate_one_question(
        client: AsyncOpenAI,
        item: Dict,
        idx: int,
        total: int,
        model_name: str,
        api_type: str,
        cutoff_date: datetime,
        model_sem: asyncio.Semaphore,
        pbar: tqdm,
        timeout: int
) -> Dict:
    """Evaluate one CLOZE question with comprehensive metrics"""

    title = item.get("title", "N/A")
    cloze = item.get("cloze_question", "")
    date_str = item.get("date", "1900-01-01")
    link = item.get("link", "")
    section = item.get("section", "abstract")
    num_blanks = item.get("num_blanks", 5)

    # Extract correct answers
    correct_answers = []
    for i in range(1, num_blanks + 1):
        ans = item.get(f"answer{i}", "")
        correct_answers.append(ans)

    # Skip error entries
    if cloze == "ERROR":
        pbar.update(1)
        return None

    try:
        # Get model's answers
        model_answers = await ask_model(
            client, title, link, cloze, num_blanks, 
            model_name, api_type, model_sem, timeout
        )

        # Compute all metrics for each blank
        blank_metrics = []
        for ref, hyp in zip(correct_answers, model_answers):
            metrics = compute_all_metrics_for_blank(ref, hyp)
            blank_metrics.append(metrics)

        # Aggregate metrics
        exact_match_blanks = [m['exact_match'] for m in blank_metrics]
        f1_scores = [m['f1_score'] for m in blank_metrics]
        rouge_scores = [m['rouge_l'] for m in blank_metrics]
        bleu_scores = [m['bleu'] for m in blank_metrics]
        binary_scores = [m['binary_correct'] for m in blank_metrics]

        # Total metrics
        total_exact_match = 1 if all(exact_match_blanks) else 0
        total_binary_score = sum(binary_scores)
        avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else 0
        avg_rouge_score = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0

        # Parse date
        paper_date = parse_date(date_str)
        before_cutoff = paper_date < cutoff_date

        pbar.update(1)

        return {
            "link": link,
            "title": title,
            "date": date_str,
            "section": section,
            "before_cutoff": before_cutoff,
            "cloze_question": cloze,
            "correct_answers": correct_answers,
            "model_answers": model_answers,
            # Per-blank metrics
            "blank_exact_match": exact_match_blanks,
            "blank_f1_scores": f1_scores,
            "blank_rouge_scores": rouge_scores,
            "blank_bleu_scores": bleu_scores,
            "blank_binary_scores": binary_scores,
            # Aggregated metrics
            "total_exact_match": total_exact_match,
            "total_binary_score": total_binary_score,
            "avg_f1_score": avg_f1_score,
            "avg_rouge_score": avg_rouge_score,
            "avg_bleu_score": avg_bleu_score
        }

    except Exception as e:
        tqdm.write(f"[{idx}/{total}] ❌ Evaluation error: {str(e)[:60]}")
        pbar.update(1)
        return None


async def run_evaluation(
        model_config: Dict,
        input_file: str,
        output_dir: Path,
        model_concurrency: int,
        api_timeout: int
):
    """Run full evaluation for a specific model"""

    print(f"Loading CLOZE questions from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    valid_data = [item for item in data if item.get("cloze_question") != "ERROR"]
    total = len(valid_data)
    skipped = len(data) - total

    print(f"Total entries: {len(data)}")
    print(f"Valid questions: {total}")
    if skipped > 0:
        print(f"Skipped errors: {skipped}")
    print()

    # Create clients based on API type
    if model_config["api"] == "openai":
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        model_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    else:  # openrouter
        if not OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        model_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )

    model_sem = asyncio.Semaphore(model_concurrency)

    # Create progress bar
    with tqdm(total=total, desc=f"Evaluating {model_config['display_name']}", unit="question", colour="green") as pbar:
        tasks = [
            evaluate_one_question(
                model_client,
                item,
                idx,
                total,
                model_config["name"],
                model_config["api"],
                model_config["cutoff_date"],
                model_sem,
                pbar,
                api_timeout
            )
            for idx, item in enumerate(valid_data, 1)
        ]

        print("Starting evaluation...\n")
        results = await asyncio.gather(*tasks, return_exceptions=True)

    await model_client.close()

    # Filter out exceptions and None values
    valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    failed_results = [r for r in results if isinstance(r, Exception)]

    if failed_results:
        print(f"\n⚠️  {len(failed_results)} evaluation tasks failed with exceptions")

    return valid_results


# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

def calculate_comprehensive_scores(results: List[Dict]) -> Dict:
    """Calculate comprehensive scores before and after cutoff"""

    before = [r for r in results if r["before_cutoff"]]
    after = [r for r in results if not r["before_cutoff"]]

    def compute_category_stats(category_results):
        if not category_results:
            return {
                "questions": 0,
                "exact_match_questions": 0,
                "exact_match_questions_pp": 0,
                "exact_match_blanks": 0,
                "exact_match_blanks_max": 0,
                "exact_match_blanks_pp": 0,
                "binary_points": 0,
                "binary_max_points": 0,
                "binary_pp": 0,
                "avg_f1_score": 0,
                "per_blank_avg_f1": 0,
                "avg_rouge_l": 0,
                "rouge_l_pp": 0,
                "avg_bleu": 0,
                "bleu_pp": 0
            }
        
        n_questions = len(category_results)
        n_blanks = sum(len(r["blank_exact_match"]) for r in category_results)
        
        exact_match_questions = sum(r["total_exact_match"] for r in category_results)
        exact_match_blanks = sum(sum(r["blank_exact_match"]) for r in category_results)
        
        binary_points = sum(r["total_binary_score"] for r in category_results)
        binary_max = n_blanks
        
        avg_f1 = sum(r["avg_f1_score"] for r in category_results) / n_questions
        per_blank_f1 = sum(sum(r["blank_f1_scores"]) for r in category_results) / n_blanks
        
        avg_rouge = sum(r["avg_rouge_score"] for r in category_results) / n_questions
        avg_bleu = sum(r["avg_bleu_score"] for r in category_results) / n_questions
        
        return {
            "questions": n_questions,
            "exact_match_questions": exact_match_questions,
            "exact_match_questions_pp": (exact_match_questions / n_questions * 100) if n_questions > 0 else 0,
            "exact_match_blanks": exact_match_blanks,
            "exact_match_blanks_max": n_blanks,
            "exact_match_blanks_pp": (exact_match_blanks / n_blanks * 100) if n_blanks > 0 else 0,
            "binary_points": binary_points,
            "binary_max_points": binary_max,
            "binary_pp": (binary_points / binary_max * 100) if binary_max > 0 else 0,
            "avg_f1_score": avg_f1,
            "per_blank_avg_f1": per_blank_f1,
            "avg_rouge_l": avg_rouge,
            "rouge_l_pp": avg_rouge * 100,
            "avg_bleu": avg_bleu,
            "bleu_pp": avg_bleu * 100
        }
    
    return {
        "before_cutoff": compute_category_stats(before),
        "after_cutoff": compute_category_stats(after),
        "overall": compute_category_stats(results)
    }


def print_comprehensive_results(model_display_name: str, cutoff_date: datetime, scores: Dict):
    """Print comprehensive evaluation results for a model"""
    print("\n" + "=" * 80)
    print(f"COMPREHENSIVE RESULTS: {model_display_name}")
    print("=" * 80)

    before = scores['before_cutoff']
    after = scores['after_cutoff']
    overall = scores['overall']

    cutoff_str = cutoff_date.strftime("%Y-%m-%d")

    # Summary table
    print(f"\n{'Metric':<30} {'Pre-Cutoff':<15} {'Post-Cutoff':<15} {'Overall':<15} {'Gap (pp)'}")
    print("-" * 80)
    
    print(f"{'Questions':<30} {before['questions']:<15} {after['questions']:<15} {overall['questions']:<15}")
    print()
    
    # Exact Match metrics
    print(f"{'Exact Match (Questions):':<30} {before['exact_match_questions_pp']:>10.2f} pp    {after['exact_match_questions_pp']:>10.2f} pp    {overall['exact_match_questions_pp']:>10.2f} pp    {before['exact_match_questions_pp'] - after['exact_match_questions_pp']:>+7.2f}")
    print(f"{'  (count/total)':<30} {before['exact_match_questions']}/{before['questions']:<10} {after['exact_match_questions']}/{after['questions']:<10} {overall['exact_match_questions']}/{overall['questions']:<10}")
    print()
    
    print(f"{'Exact Match (Blanks):':<30} {before['exact_match_blanks_pp']:>10.2f} pp    {after['exact_match_blanks_pp']:>10.2f} pp    {overall['exact_match_blanks_pp']:>10.2f} pp    {before['exact_match_blanks_pp'] - after['exact_match_blanks_pp']:>+7.2f}")
    print(f"{'  (count/total)':<30} {before['exact_match_blanks']}/{before['exact_match_blanks_max']:<10} {after['exact_match_blanks']}/{after['exact_match_blanks_max']:<10} {overall['exact_match_blanks']}/{overall['exact_match_blanks_max']:<10}")
    print()
    
    # Binary score
    print(f"{'Binary Score (ROUGE+BLEU):':<30} {before['binary_pp']:>10.2f} pp    {after['binary_pp']:>10.2f} pp    {overall['binary_pp']:>10.2f} pp    {before['binary_pp'] - after['binary_pp']:>+7.2f}")
    print(f"{'  (points/total)':<30} {before['binary_points']}/{before['binary_max_points']:<10} {after['binary_points']}/{after['binary_max_points']:<10} {overall['binary_points']}/{overall['binary_max_points']:<10}")
    print()
    
    # F1 scores
    print(f"{'F1 Score (per question):':<30} {before['avg_f1_score']:>10.4f}      {after['avg_f1_score']:>10.4f}      {overall['avg_f1_score']:>10.4f}      {before['avg_f1_score'] - after['avg_f1_score']:>+7.4f}")
    print(f"{'F1 Score (per blank):':<30} {before['per_blank_avg_f1']:>10.4f}      {after['per_blank_avg_f1']:>10.4f}      {overall['per_blank_avg_f1']:>10.4f}      {before['per_blank_avg_f1'] - after['per_blank_avg_f1']:>+7.4f}")
    print()
    
    # ROUGE-L scores
    print(f"{'ROUGE-L:':<30} {before['rouge_l_pp']:>10.2f} pp    {after['rouge_l_pp']:>10.2f} pp    {overall['rouge_l_pp']:>10.2f} pp    {before['rouge_l_pp'] - after['rouge_l_pp']:>+7.2f}")
    print()
    
    # BLEU scores
    print(f"{'BLEU:':<30} {before['bleu_pp']:>10.2f} pp    {after['bleu_pp']:>10.2f} pp    {overall['bleu_pp']:>10.2f} pp    {before['bleu_pp'] - after['bleu_pp']:>+7.2f}")
    
    print("=" * 80)


def save_results(results: List[Dict], scores: Dict, model_config: Dict, output_dir: Path):
    """Save evaluation results and scores to JSON files"""
    
    # Save detailed results
    results_file = output_dir / f"evaluation_{model_config['output_prefix']}_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"💾 Detailed results: {results_file}")

    # Save scores
    scores_file = output_dir / f"evaluation_{model_config['output_prefix']}_scores.json"
    with open(scores_file, 'w', encoding='utf-8') as f:
        json.dump(scores, f, indent=2)
    print(f"💾 Score summary: {scores_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Evaluate CLOZE questions with comprehensive metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with default models
  python evaluateCLOZE.py -i cloze_questions.json
  
  # Evaluate specific model
  python evaluateCLOZE.py -i cloze_questions.json -m gpt-4o-mini --cutoff 2023-10-01
  
  # Custom concurrency and timeout
  python evaluateCLOZE.py -i cloze_questions.json -c 100 -t 180
  
  # Save to specific directory
  python evaluateCLOZE.py -i cloze_questions.json -o ./results/
        """
    )
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True,
                        help='Input JSON file with CLOZE questions')
    
    # Model configuration
    parser.add_argument('-m', '--model',
                        help='Specific model to evaluate (e.g., gpt-4o-mini). If not specified, evaluates all default models.')
    parser.add_argument('--api', choices=['openai', 'openrouter'],
                        help='API type (required if --model is specified)')
    parser.add_argument('--cutoff',
                        help='Cutoff date (YYYY-MM-DD) for pre/post analysis (required if --model is specified)')
    parser.add_argument('--display-name',
                        help='Display name for the model (optional, defaults to model name)')
    
    # Output configuration
    parser.add_argument('-o', '--output-dir', default='.',
                        help='Output directory for results (default: current directory)')
    
    # API configuration
    parser.add_argument('-c', '--concurrency', type=int, default=DEFAULT_MODEL_CONCURRENCY,
                        help=f'Number of concurrent API calls (default: {DEFAULT_MODEL_CONCURRENCY})')
    parser.add_argument('-t', '--timeout', type=int, default=DEFAULT_API_TIMEOUT,
                        help=f'API timeout in seconds (default: {DEFAULT_API_TIMEOUT})')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        print(f"❌ Error: Input file '{args.input}' not found")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine models to evaluate
    if args.model:
        # Single model evaluation
        if not args.api:
            print("❌ Error: --api is required when specifying a custom model")
            return
        if not args.cutoff:
            print("❌ Error: --cutoff is required when specifying a custom model")
            return
        
        try:
            cutoff_date = datetime.strptime(args.cutoff, "%Y-%m-%d")
        except ValueError:
            print(f"❌ Error: Invalid cutoff date format. Use YYYY-MM-DD")
            return
        
        models_to_eval = [{
            "name": args.model,
            "display_name": args.display_name or args.model,
            "api": args.api,
            "cutoff_date": cutoff_date,
            "output_prefix": args.model.replace('/', '_').replace('-', '_')
        }]
    else:
        # Use default models
        models_to_eval = DEFAULT_MODELS
    
    # Print configuration
    print("=" * 80)
    print("CLOZE EVALUATION CONFIGURATION")
    print("=" * 80)
    print(f"Input file:       {args.input}")
    print(f"Output directory: {output_dir}")
    print(f"Models:           {len(models_to_eval)}")
    print(f"Concurrency:      {args.concurrency}")
    print(f"Timeout:          {args.timeout}s")
    print("=" * 80)
    print()
    
    # Evaluate each model
    all_model_results = {}
    
    for model_idx, model_config in enumerate(models_to_eval, 1):
        print("\n" + "=" * 80)
        print(f"EVALUATING MODEL {model_idx}/{len(models_to_eval)}: {model_config['display_name']}")
        print(f"  API: {model_config['api']}")
        print(f"  Cutoff: {model_config['cutoff_date'].strftime('%Y-%m-%d')}")
        print("=" * 80)
        print()
        
        # Run evaluation
        eval_results = asyncio.run(run_evaluation(
            model_config,
            args.input,
            output_dir,
            args.concurrency,
            args.timeout
        ))
        
        # Calculate comprehensive scores
        scores = calculate_comprehensive_scores(eval_results)
        
        # Save results
        save_results(eval_results, scores, model_config, output_dir)
        
        # Print results
        print_comprehensive_results(
            model_config['display_name'],
            model_config['cutoff_date'],
            scores
        )
        
        # Store for comparison
        all_model_results[model_config['display_name']] = {
            'config': model_config,
            'scores': scores
        }
    
    # Print comparison if multiple models
    if len(models_to_eval) > 1:
        print("\n\n" + "=" * 80)
        print("COMPARISON SUMMARY: ALL MODELS")
        print("=" * 80)
        print()
        
        # Exact Match comparison
        print("Exact Match (Questions) Performance:")
        print(f"{'Model':<25} {'Pre-Cutoff':<15} {'Post-Cutoff':<15} {'Gap (pp)'}")
        print("-" * 80)
        
        for model_name, data in all_model_results.items():
            scores = data['scores']
            before = scores['before_cutoff']
            after = scores['after_cutoff']
            gap = before['exact_match_questions_pp'] - after['exact_match_questions_pp']
            print(f"{model_name:<25} {before['exact_match_questions_pp']:>10.2f} pp    {after['exact_match_questions_pp']:>10.2f} pp      {gap:>+7.2f}")
        
        print()
        
        # Binary Score comparison
        print("Binary Score Performance:")
        print(f"{'Model':<25} {'Pre-Cutoff':<15} {'Post-Cutoff':<15} {'Gap (pp)'}")
        print("-" * 80)
        
        for model_name, data in all_model_results.items():
            scores = data['scores']
            before = scores['before_cutoff']
            after = scores['after_cutoff']
            gap = before['binary_pp'] - after['binary_pp']
            print(f"{model_name:<25} {before['binary_pp']:>10.2f} pp    {after['binary_pp']:>10.2f} pp      {gap:>+7.2f}")
        
        print()
        
        # F1 Score comparison
        print("F1 Score (per question) Performance:")
        print(f"{'Model':<25} {'Pre-Cutoff':<15} {'Post-Cutoff':<15} {'Gap'}")
        print("-" * 80)
        
        for model_name, data in all_model_results.items():
            scores = data['scores']
            before = scores['before_cutoff']
            after = scores['after_cutoff']
            gap = before['avg_f1_score'] - after['avg_f1_score']
            print(f"{model_name:<25} {before['avg_f1_score']:>10.4f}      {after['avg_f1_score']:>10.4f}      {gap:>+7.4f}")
        
        print()
        
        # ROUGE-L comparison
        print("ROUGE-L Performance:")
        print(f"{'Model':<25} {'Pre-Cutoff':<15} {'Post-Cutoff':<15} {'Gap (pp)'}")
        print("-" * 80)
        
        for model_name, data in all_model_results.items():
            scores = data['scores']
            before = scores['before_cutoff']
            after = scores['after_cutoff']
            gap = before['rouge_l_pp'] - after['rouge_l_pp']
            print(f"{model_name:<25} {before['rouge_l_pp']:>10.2f} pp    {after['rouge_l_pp']:>10.2f} pp      {gap:>+7.2f}")
        
        print()
        
        # BLEU comparison
        print("BLEU Performance:")
        print(f"{'Model':<25} {'Pre-Cutoff':<15} {'Post-Cutoff':<15} {'Gap (pp)'}")
        print("-" * 80)
        
        for model_name, data in all_model_results.items():
            scores = data['scores']
            before = scores['before_cutoff']
            after = scores['after_cutoff']
            gap = before['bleu_pp'] - after['bleu_pp']
            print(f"{model_name:<25} {before['bleu_pp']:>10.2f} pp    {after['bleu_pp']:>10.2f} pp      {gap:>+7.2f}")
        
        print()
        print("=" * 80)
    
    print("\n✅ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
