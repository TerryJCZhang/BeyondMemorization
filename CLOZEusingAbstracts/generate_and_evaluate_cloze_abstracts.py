#!/usr/bin/env python3
"""
COMBINED: Generate and Evaluate CLOZE questions from ABSTRACTS
Part 1: Generate ONE abstract with FIVE blanks [blank1] through [blank5]
        Each blank: max 5 words, self-contained, meaningful expression
Part 2: Evaluate models on the generated CLOZE questions
        Metrics: Semantic equivalence (binary) + ROUGE-L + BLEU scores
"""

import asyncio
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple

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
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Configuration
CUTOFF_DATE_OPENAI = datetime(2023, 10, 1)
CUTOFF_DATE_LLAMA = datetime(2023, 7, 1)  # Llama 3.1 405B cutoff
CUTOFF_DATE_CLAUDE = datetime(2024, 4, 1)  # Claude 3.5 Sonnet cutoff

MODEL_CONCURRENCY = 50
JUDGE_CONCURRENCY = 100
GENERATION_CONCURRENCY = 700

# Timeout settings (seconds)
API_TIMEOUT = 120  # 2 minutes timeout for API calls
JUDGE_TIMEOUT = 60  # 1 minute timeout for judge calls

# Model configurations
MODELS_TO_EVAL = [
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

# Global lock for file writing
_file_write_lock = None


def get_file_lock():
    """Get or create file write lock"""
    global _file_write_lock
    if _file_write_lock is None:
        _file_write_lock = asyncio.Lock()
    return _file_write_lock


# ============================================================================
# PART 1: GENERATION
# ============================================================================

async def save_result_to_file(output_file: str, result: Dict):
    """Safely append result to JSON file (with lock protection)"""
    lock = get_file_lock()
    async with lock:
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data.append(result)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            tqdm.write(f"  ❌ File write error: {e}")


def create_user_prompt(link: str, title: str, abstract: str, date: str) -> str:
    """Create user prompt for generating ONE abstract with 5 blanks"""
    return (
        "Create ONE CLOZE version of this abstract with EXACTLY 5 BLANKS.\n\n"
        "CRITICAL REQUIREMENTS:\n"
        "1. Each answer MUST be 1-6 WORDS (strictly enforced)\n"
        "2. Answers must be WORDS or EXPRESSIONS - NOT formulas, NOT equations, NOT mathematical notation\n"
        "3. NEVER use single character (x, n, k) or single number (5, 10) as answer\n"
        "4. Good examples: 'harmonic centrality', 'graph products', 'geodesic distances', 'Cartesian product'\n"
        "5. Bad examples: 'x', '5', 'P_2', 'f(x)', '$\\\\lambda$', formulas, equations\n\n"
        "BLANKING RULES:\n"
        "1. Use [blank1], [blank2], [blank3], [blank4], [blank5]\n"
        "2. Never blank: prepositions, pronouns, articles, conjunctions\n"
        "3. Always blank: meaningful concepts, methods, results that test recall from memory\n\n"
        "OUTPUT JSON:\n"
        '{"question": "<abstract with blanks>", "answer1": "...", "answer2": "...", "answer3": "...", "answer4": "...", "answer5": "..."}\n\n'
        f"Link: {link}\n"
        f"Title: {title}\n"
        f"Date: {date}\n"
        f"Abstract: {abstract}"
    )


def validate_answer_length(answer: str) -> bool:
    """Check if answer is 1-9 words"""
    word_count = len(answer.split())
    return 1 <= word_count <= 9


async def generate_single_cloze(
        client: AsyncOpenAI,
        item: Dict,
        idx: int,
        total: int,
        semaphore: asyncio.Semaphore,
        output_file: str,
        pbar: tqdm,
        max_retries: int = 3
) -> Dict:
    """Generate ONE CLOZE question with 5 blanks for a single abstract"""

    link = item.get('link', 'N/A')
    title = item.get('title', 'N/A')
    abstract = item.get('abstract', '')
    date = item.get('date', 'N/A')

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            try:
                user_prompt = create_user_prompt(link, title, abstract, date)
                system_prompt = (
                    "You are a meticulous scientific editor who writes graduate-level CLOZE questions "
                    "from research abstracts in physics and mathematics. Always follow the user's "
                    "instructions exactly and return valid JSON."
                )
                full_prompt = f"{system_prompt}\n\n{user_prompt}"

                # Call OpenAI Responses API with timeout
                response = await asyncio.wait_for(
                    client.responses.create(
                        model="gpt-5-mini",
                        input=full_prompt,
                        reasoning={"effort": "low"},
                        text={
                            "verbosity": "low",
                            "format": {"type": "json_object"}
                        },
                        max_output_tokens=2500
                    ),
                    timeout=API_TIMEOUT
                )

                # Parse response
                output_text = ""
                if response and response.output:
                    for item_output in response.output:
                        if hasattr(item_output, "content") and item_output.content:
                            for content in item_output.content:
                                if hasattr(content, "text"):
                                    output_text += content.text

                if not output_text:
                    raise ValueError("No text in response")

                cloze_data = json.loads(output_text)

                # Validate format - must have question and 5 answers
                required_keys = ['question', 'answer1', 'answer2', 'answer3', 'answer4', 'answer5']
                if not all(key in cloze_data for key in required_keys):
                    raise ValueError("Missing required keys in response")

                question = cloze_data['question'].strip()
                answers = [
                    cloze_data['answer1'].strip(),
                    cloze_data['answer2'].strip(),
                    cloze_data['answer3'].strip(),
                    cloze_data['answer4'].strip(),
                    cloze_data['answer5'].strip()
                ]

                # Validate not empty
                if not question or any(not ans for ans in answers):
                    raise ValueError("Empty question or answer")

                # Verify question contains all 5 blanks
                for i in range(1, 6):
                    if f'[blank{i}]' not in question.lower():
                        raise ValueError(f"Question missing [blank{i}] marker")

                # Validate answers are not single characters
                for i, ans in enumerate(answers, 1):
                    if len(ans) == 1:
                        raise ValueError(f"Answer {i} is too short (single character): '{ans}'")

                # Validate answer length (1-6 words)
                for i, ans in enumerate(answers, 1):
                    if not validate_answer_length(ans):
                        word_count = len(ans.split())
                        raise ValueError(f"Answer {i} has {word_count} words (must be 1-6): '{ans}'")

                # Success!
                result = {
                    "link": link,
                    "title": title,
                    "date": date,
                    "abstract": abstract,
                    "cloze_question": question,
                    "answer1": answers[0],
                    "answer2": answers[1],
                    "answer3": answers[2],
                    "answer4": answers[3],
                    "answer5": answers[4],
                    "error": None
                }

                await save_result_to_file(output_file, result)
                pbar.update(1)
                return result

            except asyncio.TimeoutError:
                tqdm.write(f"[{idx}/{total}] ⏱️  Timeout (attempt {attempt}/{max_retries})")
                if attempt < max_retries:
                    await asyncio.sleep(2)
                    continue
            except json.JSONDecodeError as e:
                tqdm.write(f"[{idx}/{total}] ❌ JSON error (attempt {attempt}): {str(e)[:60]}")
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                tqdm.write(f"[{idx}/{total}] ❌ Error (attempt {attempt}): {str(e)[:60]}")
                if attempt < max_retries:
                    await asyncio.sleep(1)
                    continue

        # All retries failed
        tqdm.write(f"[{idx}/{total}] 🚨 Failed after {max_retries} attempts: {title[:50]}")
        error_result = {
            "link": link,
            "title": title,
            "date": date,
            "abstract": abstract,
            "cloze_question": "ERROR",
            "answer1": "ERROR",
            "answer2": "ERROR",
            "answer3": "ERROR",
            "answer4": "ERROR",
            "answer5": "ERROR",
            "error": "All retries failed"
        }

        await save_result_to_file(output_file, error_result)
        pbar.update(1)
        return error_result


async def process_abstracts(input_file: str = "realmath_processed.json"):
    """Process ALL abstracts - NO LIMIT"""

    print("Loading data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    print(f"Total records: {total}")
    print(f"Processing: ALL {total} records\n")

    # Initialize output file
    output_file = "realmath_abstracts_with_cloze.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([], f)

    print(f"Output: {output_file}\n")

    # Create client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(GENERATION_CONCURRENCY)

    # Create progress bar
    with tqdm(total=total, desc="Generating CLOZE questions", unit="question") as pbar:
        # Create tasks
        tasks = [
            generate_single_cloze(client, item, idx, total, semaphore, output_file, pbar)
            for idx, item in enumerate(data, 1)
        ]

        print("Starting generation...\n")
        results = await asyncio.gather(*tasks, return_exceptions=True)

    await client.close()

    # Filter out exceptions
    valid_results = [r for r in results if not isinstance(r, Exception)]
    failed_results = [r for r in results if isinstance(r, Exception)]

    if failed_results:
        print(f"\n⚠️  {len(failed_results)} tasks failed with exceptions")

    return valid_results


# ============================================================================
# PART 2: EVALUATION
# ============================================================================

def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime"""
    try:
        return datetime.strptime(date_str[:10], "%Y-%m-%d")
    except:
        return datetime(1900, 1, 1)


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
    # For short answers (1-5 words), focus more on unigrams and bigrams
    weights = (0.5, 0.3, 0.15, 0.05)

    try:
        score = sentence_bleu([reference_tokens], hypothesis_tokens,
                              weights=weights, smoothing_function=smoothing)
        return score
    except:
        return 0.0


async def ask_model(
        client: AsyncOpenAI,
        title: str,
        link: str,
        cloze_question: str,
        model_name: str,
        api_type: str,
        semaphore: asyncio.Semaphore
) -> List[str]:
    """Ask model to recall paper and fill 5 blanks"""
    async with semaphore:
        try:
            prompt = (
                f"Title: {title}\n"
                f"Link: {link}\n\n"
                f"First, try to recall this paper from your training data based on the title.\n"
                f"Then, fill in the blanks to the best of your memory.\n\n"
                f"IMPORTANT:\n"
                f"- Answer from MEMORY of what you learned during training\n"
                f"- Each answer: 1-6 words maximum\n"
                f"- If you don't remember, make your best educated guess based on the context\n\n"
                f"Abstract with blanks:\n{cloze_question}\n\n"
                "Provide your answers:\n"
                "blank1: <answer>\n"
                "blank2: <answer>\n"
                "blank3: <answer>\n"
                "blank4: <answer>\n"
                "blank5: <answer>"
            )

            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=300
                ),
                timeout=API_TIMEOUT
            )

            content = response.choices[0].message.content.strip()

            # Parse answers
            answers = []
            for i in range(1, 6):
                marker = f"blank{i}:"
                if marker in content.lower():
                    start = content.lower().index(marker) + len(marker)
                    rest = content[start:].strip()
                    end = len(rest)
                    for j in range(i + 1, 6):
                        next_marker = f"blank{j}:"
                        if next_marker in rest.lower():
                            end = min(end, rest.lower().index(next_marker))
                    answer = rest[:end].strip().split('\n')[0].strip()
                    answers.append(answer)
                else:
                    answers.append("NO_ANSWER")

            return answers if len(answers) == 5 else ["NO_ANSWER"] * 5

        except asyncio.TimeoutError:
            tqdm.write(f"  ⏱️  Model timeout for: {title[:50]}")
            return ["TIMEOUT"] * 5
        except Exception as e:
            tqdm.write(f"  ❌ Model error: {str(e)[:60]}")
            return ["ERROR"] * 5


async def judge_one_blank(
        client: AsyncOpenAI,
        abstract: str,
        cloze_question: str,
        blank_num: int,
        correct_answer: str,
        model_answer: str,
        semaphore: asyncio.Semaphore
) -> Tuple[bool, int, float, float]:
    """Judge one blank: binary + ROUGE-L + BLEU"""
    rouge_score = calculate_rouge_l(correct_answer.lower(), model_answer.lower())
    bleu_score = calculate_bleu(correct_answer.lower(), model_answer.lower())

    async with semaphore:
        try:
            prompt = (
                "You are evaluating a CLOZE question designed to test paper memorization.\n\n"
                f"Original Abstract:\n{abstract}\n\n"
                f"CLOZE Question:\n{cloze_question}\n\n"
                f"Evaluation for [blank{blank_num}]:\n"
                f"  Correct answer: {correct_answer}\n"
                f"  Model answer: {model_answer}\n\n"
                "Task: Determine if the model correctly recalled the concept from memory.\n"
                "- Consider synonyms and semantically equivalent terms as CORRECT\n"
                "- The context is the abstract above\n"
                "- Focus on whether the model remembered the right concept\n\n"
                "Respond with ONLY: 1 (correct/equivalent) or 0 (incorrect/different)"
            )

            response = await asyncio.wait_for(
                client.responses.create(
                    model="gpt-5-mini",
                    input=prompt,
                    reasoning={"effort": "low"},
                    text={"verbosity": "low"},
                    max_output_tokens=200
                ),
                timeout=JUDGE_TIMEOUT
            )

            output_text = ""
            if response and response.output:
                for item in response.output:
                    if hasattr(item, "content") and item.content:
                        for content in item.content:
                            if hasattr(content, "text"):
                                output_text += content.text

            is_correct = "1" in output_text
            binary_score = 1 if is_correct else 0

            return is_correct, binary_score, rouge_score, bleu_score

        except asyncio.TimeoutError:
            tqdm.write(f"  ⏱️  Judge timeout for blank{blank_num}")
            return False, 0, rouge_score, bleu_score
        except Exception as e:
            tqdm.write(f"  ❌ Judge error blank{blank_num}: {str(e)[:60]}")
            return False, 0, rouge_score, bleu_score


async def evaluate_one_question(
        client: AsyncOpenAI,
        judge_client: AsyncOpenAI,
        item: Dict,
        idx: int,
        total: int,
        model_name: str,
        api_type: str,
        cutoff_date: datetime,
        model_sem: asyncio.Semaphore,
        judge_sem: asyncio.Semaphore,
        pbar: tqdm
) -> Dict:
    """Evaluate one ABSTRACT-based CLOZE question (with 5 blanks)"""

    title = item.get("title", "N/A")
    cloze = item.get("cloze_question", "")
    abstract = item.get("abstract", "")
    date_str = item.get("date", "1900-01-01")
    link = item.get("link", "")

    correct_answers = [
        item.get("answer1", ""),
        item.get("answer2", ""),
        item.get("answer3", ""),
        item.get("answer4", ""),
        item.get("answer5", "")
    ]

    # Skip error entries
    if cloze == "ERROR":
        pbar.update(1)
        return None

    try:
        # Step 1: Get model's 5 answers (with title + link for recall)
        model_answers = await ask_model(client, title, link, cloze, model_name, api_type, model_sem)

        # Step 2: Judge each of the 5 blanks (semantic + ROUGE-L + BLEU)
        judge_tasks = [
            judge_one_blank(judge_client, abstract, cloze, i + 1, correct_answers[i], model_answers[i], judge_sem)
            for i in range(5)
        ]
        results = await asyncio.gather(*judge_tasks)

        # Calculate scores
        blank_scores = [score for _, score, _, _ in results]
        rouge_scores = [rouge for _, _, rouge, _ in results]
        bleu_scores = [bleu for _, _, _, bleu in results]

        total_binary_score = sum(blank_scores)
        avg_rouge_score = sum(rouge_scores) / len(rouge_scores)
        avg_bleu_score = sum(bleu_scores) / len(bleu_scores)

        # Parse date
        paper_date = parse_date(date_str)
        before_cutoff = paper_date < cutoff_date

        pbar.update(1)

        return {
            "link": link,
            "title": title,
            "date": date_str,
            "before_cutoff": before_cutoff,
            "cloze_question": cloze,
            "correct_answers": correct_answers,
            "model_answers": model_answers,
            "blank_binary_scores": blank_scores,
            "blank_rouge_scores": rouge_scores,
            "blank_bleu_scores": bleu_scores,
            "total_binary_score": total_binary_score,
            "avg_rouge_score": avg_rouge_score,
            "avg_bleu_score": avg_bleu_score
        }

    except Exception as e:
        tqdm.write(f"[{idx}/{total}] ❌ Evaluation error: {str(e)[:60]}")
        pbar.update(1)
        return None


async def run_evaluation(
        model_config: Dict,
        input_file: str = "realmath_abstracts_with_cloze.json"
):
    """Run full evaluation for a specific model"""

    print(f"Loading CLOZE questions...")
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
        model_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    else:  # openrouter
        model_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY
        )

    # Judge always uses OpenAI
    judge_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    model_sem = asyncio.Semaphore(MODEL_CONCURRENCY)
    judge_sem = asyncio.Semaphore(JUDGE_CONCURRENCY)

    # Create progress bar
    with tqdm(total=total, desc=f"Evaluating {model_config['display_name']}", unit="question") as pbar:
        tasks = [
            evaluate_one_question(
                model_client,
                judge_client,
                item,
                idx,
                total,
                model_config["name"],
                model_config["api"],
                model_config["cutoff_date"],
                model_sem,
                judge_sem,
                pbar
            )
            for idx, item in enumerate(valid_data, 1)
        ]

        print("Starting evaluation...\n")
        results = await asyncio.gather(*tasks, return_exceptions=True)

    await model_client.close()
    await judge_client.close()

    # Filter out exceptions and None values
    valid_results = [r for r in results if r is not None and not isinstance(r, Exception)]
    failed_results = [r for r in results if isinstance(r, Exception)]

    if failed_results:
        print(f"\n⚠️  {len(failed_results)} evaluation tasks failed with exceptions")

    return valid_results


def calculate_scores(results: List[Dict]) -> Dict:
    """Calculate scores before and after cutoff (binary, ROUGE-L, and BLEU) in percentage points"""

    before = [r for r in results if r["before_cutoff"]]
    after = [r for r in results if not r["before_cutoff"]]

    # Binary scores (total points and max possible points)
    before_points = sum(r["total_binary_score"] for r in before)
    before_max = len(before) * 5
    after_points = sum(r["total_binary_score"] for r in after)
    after_max = len(after) * 5
    total_points = before_points + after_points
    total_max = len(results) * 5

    # Binary percentages (already in percentage points)
    before_binary_pp = (before_points / before_max * 100) if before_max > 0 else 0
    after_binary_pp = (after_points / after_max * 100) if after_max > 0 else 0
    overall_binary_pp = (total_points / total_max * 100) if total_max > 0 else 0

    # ROUGE-L scores (convert to percentage points)
    before_rouge = (sum(r["avg_rouge_score"] for r in before) / len(before)) if before else 0
    after_rouge = (sum(r["avg_rouge_score"] for r in after) / len(after)) if after else 0
    overall_rouge = (sum(r["avg_rouge_score"] for r in results) / len(results)) if results else 0

    before_rouge_pp = before_rouge * 100
    after_rouge_pp = after_rouge * 100
    overall_rouge_pp = overall_rouge * 100

    # BLEU scores (convert to percentage points)
    before_bleu = (sum(r["avg_bleu_score"] for r in before) / len(before)) if before else 0
    after_bleu = (sum(r["avg_bleu_score"] for r in after) / len(after)) if after else 0
    overall_bleu = (sum(r["avg_bleu_score"] for r in results) / len(results)) if results else 0

    before_bleu_pp = before_bleu * 100
    after_bleu_pp = after_bleu * 100
    overall_bleu_pp = overall_bleu * 100

    return {
        "before_cutoff": {
            "questions": len(before),
            "binary_points": before_points,
            "binary_max_points": before_max,
            "binary_pp": before_binary_pp,
            "rouge_l_pp": before_rouge_pp,
            "bleu_pp": before_bleu_pp
        },
        "after_cutoff": {
            "questions": len(after),
            "binary_points": after_points,
            "binary_max_points": after_max,
            "binary_pp": after_binary_pp,
            "rouge_l_pp": after_rouge_pp,
            "bleu_pp": after_bleu_pp
        },
        "overall": {
            "questions": len(results),
            "binary_points": total_points,
            "binary_max_points": total_max,
            "binary_pp": overall_binary_pp,
            "rouge_l_pp": overall_rouge_pp,
            "bleu_pp": overall_bleu_pp
        }
    }


def print_model_results(model_display_name: str, cutoff_date: datetime, scores: Dict):
    """Print evaluation results for a model - all metrics in percentage points"""
    print("\n" + "=" * 70)
    print(f"RESULTS: {model_display_name}")
    print("=" * 70)

    before = scores['before_cutoff']
    after = scores['after_cutoff']

    cutoff_str = cutoff_date.strftime("%Y-%m-%d")

    # Binary Score Table
    print(f"\nBinary Score (Cutoff: {cutoff_str}):")
    print(f"{'Metric':<20} {'Pre-Cutoff':<15} {'Post-Cutoff':<15} {'Gap (pp)'}")
    print("-" * 70)

    binary_gap = before['binary_pp'] - after['binary_pp'] if before['questions'] > 0 and after['questions'] > 0 else 0

    print(
        f"{'Accuracy':<20} {before['binary_pp']:>10.2f} pp    {after['binary_pp']:>10.2f} pp      {binary_gap:>+7.2f}")
    print(f"{'Questions':<20} {before['questions']:>8}        {after['questions']:>8}")

    # ROUGE-L and BLEU Table (all in percentage points)
    print(f"\nContinuous Metrics (Cutoff: {cutoff_str}):")
    print(f"{'Metric':<20} {'Pre-Cutoff':<15} {'Post-Cutoff':<15} {'Gap (pp)'}")
    print("-" * 70)

    rouge_gap = before['rouge_l_pp'] - after['rouge_l_pp'] if before['questions'] > 0 and after['questions'] > 0 else 0
    bleu_gap = before['bleu_pp'] - after['bleu_pp'] if before['questions'] > 0 and after['questions'] > 0 else 0

    print(
        f"{'ROUGE-L':<20} {before['rouge_l_pp']:>10.2f} pp    {after['rouge_l_pp']:>10.2f} pp      {rouge_gap:>+7.2f}")
    print(f"{'BLEU':<20} {before['bleu_pp']:>10.2f} pp    {after['bleu_pp']:>10.2f} pp      {bleu_gap:>+7.2f}")

    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main function - runs generation once, then evaluates all models sequentially"""

    print("=" * 70)
    print("CLOZE PIPELINE: Generation + Multi-Model Evaluation")
    print("=" * 70)
    print()

    # PART 1: Generation (once)
    print("=" * 70)
    print("PART 1: GENERATION")
    print("  Format: 1 abstract with 5 blanks per question")
    print("  Answers: 1-6 words, meaningful expressions (NO single char/number)")
    print(f"  Concurrency: {GENERATION_CONCURRENCY}")
    print(f"  Timeout: {API_TIMEOUT}s per API call")
    print("=" * 70)
    print()

    gen_results = asyncio.run(process_abstracts())

    success = sum(1 for r in gen_results if r.get('error') is None)
    failed = len(gen_results) - success

    print("\n" + "=" * 70)
    print(f"✅ Generation Success: {success}/{len(gen_results)}")
    if failed > 0:
        print(f"❌ Generation Failed: {failed}/{len(gen_results)}")
    print(f"💾 Output: realmath_abstracts_with_cloze.json")
    print("=" * 70)
    print()

    # PAUSE FOR USER APPROVAL
    print("\n" + "=" * 70)
    print("⏸️  PAUSED: Please review the generated CLOZE questions")
    print(f"   File: realmath_abstracts_with_cloze.json")
    print(f"   Total generated: {success}")
    print("=" * 70)
    input("\n✋ Press ENTER when ready to proceed with evaluation...")
    print()

    # PART 2: Evaluate each model sequentially
    print("\n" + "=" * 70)
    print("PART 2: MULTI-MODEL EVALUATION")
    print(f"  Models: {len(MODELS_TO_EVAL)}")
    print("  Judge: GPT-5-mini (OpenAI Responses API)")
    print("  Metrics: Binary + ROUGE-L + BLEU (all in percentage points)")
    print(f"  Concurrency: {MODEL_CONCURRENCY} model, {JUDGE_CONCURRENCY} judge")
    print(f"  Timeouts: {API_TIMEOUT}s model, {JUDGE_TIMEOUT}s judge")
    print("=" * 70)
    print()

    all_model_results = {}

    for model_idx, model_config in enumerate(MODELS_TO_EVAL, 1):
        print("\n" + "=" * 70)
        print(f"EVALUATING MODEL {model_idx}/{len(MODELS_TO_EVAL)}: {model_config['display_name']}")
        print(f"  API: {model_config['api']}")
        print(f"  Cutoff: {model_config['cutoff_date'].strftime('%Y-%m-%d')}")
        print("=" * 70)
        print()

        # Run evaluation
        eval_results = asyncio.run(run_evaluation(model_config))

        # Save detailed results
        results_file = f"evaluation_{model_config['output_prefix']}_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Detailed results: {results_file}")

        # Calculate and save scores
        scores = calculate_scores(eval_results)
        scores_file = f"evaluation_{model_config['output_prefix']}_scores.json"
        with open(scores_file, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=2)
        print(f"💾 Score summary: {scores_file}")

        # Print results
        print_model_results(
            model_config['display_name'],
            model_config['cutoff_date'],
            scores
        )

        # Store for comparison
        all_model_results[model_config['display_name']] = scores

    # PART 3: Comparison summary
    print("\n\n" + "=" * 70)
    print("COMPARISON SUMMARY: ALL MODELS")
    print("=" * 70)
    print()

    # Binary Score Table
    print("Binary Score Performance Decay:")
    print(f"{'Model':<25} {'Pre-Cutoff':<15} {'Post-Cutoff':<15} {'Gap (pp)'}")
    print("-" * 70)

    for model_config in MODELS_TO_EVAL:
        model_name = model_config['display_name']
        scores = all_model_results[model_name]
        before = scores['before_cutoff']
        after = scores['after_cutoff']

        gap = before['binary_pp'] - after['binary_pp'] if before['questions'] > 0 and after['questions'] > 0 else 0

        print(f"{model_name:<25} {before['binary_pp']:>10.2f} pp    {after['binary_pp']:>10.2f} pp       {gap:>+7.2f}")

    print()

    # Continuous Metrics Table
    print("Continuous Metrics Performance Decay:")
    print(f"{'Model':<25} {'Metric':<12} {'Pre-Cutoff':<12} {'Post-Cutoff':<12} {'Gap (pp)'}")
    print("-" * 70)

    for model_config in MODELS_TO_EVAL:
        model_name = model_config['display_name']
        scores = all_model_results[model_name]
        before = scores['before_cutoff']
        after = scores['after_cutoff']

        rouge_gap = before['rouge_l_pp'] - after['rouge_l_pp'] if before['questions'] > 0 and after[
            'questions'] > 0 else 0
        bleu_gap = before['bleu_pp'] - after['bleu_pp'] if before['questions'] > 0 and after['questions'] > 0 else 0

        print(
            f"{model_name:<25} {'ROUGE-L':<12} {before['rouge_l_pp']:>8.2f} pp   {after['rouge_l_pp']:>8.2f} pp     {rouge_gap:>+7.2f}")
        print(f"{'':<25} {'BLEU':<12} {before['bleu_pp']:>8.2f} pp   {after['bleu_pp']:>8.2f} pp     {bleu_gap:>+7.2f}")
        print()

    print("=" * 70)
    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    main()