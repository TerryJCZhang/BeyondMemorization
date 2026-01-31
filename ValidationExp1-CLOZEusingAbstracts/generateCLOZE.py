#!/usr/bin/env python3
"""
Generate CLOZE questions from any section of research papers
- Customizable prompts via command-line arguments
- Flexible section selection (abstract, introduction, methodology, results, etc.)
- Configurable number of blanks
- Adjustable answer word limits
"""

import asyncio
import json
import os
import argparse
from datetime import datetime
from typing import Dict, List
from pathlib import Path

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai>=1.0.0",
#     "tqdm>=4.65.0",
# ]
# ///

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm
from tqdm import tqdm

# Get API keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set")

# Default configuration
DEFAULT_NUM_BLANKS = 5
DEFAULT_MIN_WORDS = 1
DEFAULT_MAX_WORDS = 6
DEFAULT_CONCURRENCY = 700
DEFAULT_TIMEOUT = 120  # seconds
DEFAULT_MAX_RETRIES = 3

# Default prompts
DEFAULT_SYSTEM_PROMPT = """You are an expert researcher deeply familiar with arXiv papers, and a meticulous scientific editor who writes graduate-level CLOZE questions from research papers in physics and mathematics. All questions you generate are derived from existing arXiv papers. Always follow the user's instructions exactly and return valid JSON."""

DEFAULT_USER_PROMPT_TEMPLATE = """Create ONE CLOZE version of this {section} with EXACTLY {num_blanks} BLANKS.

CRITICAL REQUIREMENTS:
1. Each answer MUST be {min_words}-{max_words} WORDS (strictly enforced)
2. Answers must be WORDS or EXPRESSIONS - NOT formulas, NOT equations, NOT mathematical notation
3. NEVER use single character (x, n, k) or single number (5, 10) as answer
4. Good examples: 'harmonic centrality', 'graph products', 'geodesic distances', 'Cartesian product'
5. Bad examples: 'x', '5', 'P_2', 'f(x)', '$\\lambda$', formulas, equations

BLANKING RULES:
1. Use {blank_markers}
2. Never blank: prepositions, pronouns, articles, conjunctions
3. Always blank: meaningful concepts, methods, results that test recall from memory

OUTPUT JSON:
{json_format}

Link: {link}
Title: {title}
Date: {date}
{section_label}: {content}"""

# Global lock for file writing
_file_write_lock = None


def get_file_lock():
    """Get or create file write lock"""
    global _file_write_lock
    if _file_write_lock is None:
        _file_write_lock = asyncio.Lock()
    return _file_write_lock


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


def create_user_prompt(
        item: Dict,
        section: str,
        num_blanks: int,
        min_words: int,
        max_words: int,
        custom_prompt: str = None
) -> str:
    """Create user prompt for generating CLOZE questions"""
    
    link = item.get('link', 'N/A')
    title = item.get('title', 'N/A')
    date = item.get('date', 'N/A')
    
    # Get content based on section
    content = item.get(section, '')
    if not content:
        # Try common alternatives
        section_alternatives = {
            'abstract': ['summary', 'abs'],
            'introduction': ['intro'],
            'methodology': ['methods', 'method'],
            'results': ['result', 'findings'],
            'conclusion': ['conclusions', 'summary']
        }
        for alt in section_alternatives.get(section, []):
            content = item.get(alt, '')
            if content:
                break
    
    if not content:
        raise ValueError(f"Section '{section}' not found in item")
    
    # Use custom prompt if provided, otherwise use default
    if custom_prompt:
        return custom_prompt.format(
            link=link,
            title=title,
            date=date,
            section=section,
            content=content,
            num_blanks=num_blanks,
            min_words=min_words,
            max_words=max_words
        )
    
    # Create blank markers and JSON format
    blank_markers = ', '.join([f'[blank{i}]' for i in range(1, num_blanks + 1)])
    json_fields = ', '.join([f'"answer{i}": "..."' for i in range(1, num_blanks + 1)])
    json_format = '{"question": "<text with blanks>", ' + json_fields + '}'
    
    return DEFAULT_USER_PROMPT_TEMPLATE.format(
        section=section,
        num_blanks=num_blanks,
        min_words=min_words,
        max_words=max_words,
        blank_markers=blank_markers,
        json_format=json_format,
        link=link,
        title=title,
        date=date,
        section_label=section.capitalize(),
        content=content
    )


def validate_answer_length(answer: str, min_words: int, max_words: int) -> bool:
    """Check if answer is within word limit"""
    word_count = len(answer.split())
    return min_words <= word_count <= max_words


async def generate_single_cloze(
        client: AsyncOpenAI,
        item: Dict,
        idx: int,
        total: int,
        semaphore: asyncio.Semaphore,
        output_file: str,
        pbar: tqdm,
        config: Dict
) -> Dict:
    """Generate ONE CLOZE question for a single item"""

    link = item.get('link', 'N/A')
    title = item.get('title', 'N/A')
    date = item.get('date', 'N/A')

    async with semaphore:
        for attempt in range(1, config['max_retries'] + 1):
            try:
                user_prompt = create_user_prompt(
                    item,
                    config['section'],
                    config['num_blanks'],
                    config['min_words'],
                    config['max_words'],
                    config.get('custom_user_prompt')
                )
                
                system_prompt = config.get('custom_system_prompt', DEFAULT_SYSTEM_PROMPT)
                full_prompt = f"{system_prompt}\n\n{user_prompt}"

                # Call OpenAI API with timeout
                response = await asyncio.wait_for(
                    client.responses.create(
                        model=config.get('model', 'gpt-4o-mini'),
                        input=full_prompt
                    ),
                    timeout=config['timeout']
                )

                # Parse response
                output_text = getattr(response, "output_text", None) or ""
                if not output_text and response and getattr(response, "output", None):
                    for item_output in response.output:
                        if hasattr(item_output, "content") and item_output.content:
                            for content in item_output.content:
                                if hasattr(content, "text") and content.text:
                                    output_text += content.text

                if not output_text:
                    raise ValueError("No text in response")

                cloze_data = json.loads(output_text)

                # Validate format - must have question and N answers
                required_keys = ['question'] + [f'answer{i}' for i in range(1, config['num_blanks'] + 1)]
                if not all(key in cloze_data for key in required_keys):
                    raise ValueError("Missing required keys in response")

                question = cloze_data['question'].strip()
                answers = [cloze_data[f'answer{i}'].strip() for i in range(1, config['num_blanks'] + 1)]

                # Validate not empty
                if not question or any(not ans for ans in answers):
                    raise ValueError("Empty question or answer")

                # Verify question contains all blanks
                for i in range(1, config['num_blanks'] + 1):
                    if f'[blank{i}]' not in question.lower():
                        raise ValueError(f"Question missing [blank{i}] marker")

                # Validate answers are not single characters
                for i, ans in enumerate(answers, 1):
                    if len(ans) == 1:
                        raise ValueError(f"Answer {i} is too short (single character): '{ans}'")

                # Validate answer length
                for i, ans in enumerate(answers, 1):
                    if not validate_answer_length(ans, config['min_words'], config['max_words']):
                        word_count = len(ans.split())
                        raise ValueError(
                            f"Answer {i} has {word_count} words (must be {config['min_words']}-{config['max_words']}): '{ans}'"
                        )

                # Success!
                result = {
                    "link": link,
                    "title": title,
                    "date": date,
                    "section": config['section'],
                    config['section']: item.get(config['section'], ''),
                    "cloze_question": question,
                    "num_blanks": config['num_blanks'],
                    "error": None
                }
                
                # Add all answers dynamically
                for i, ans in enumerate(answers, 1):
                    result[f'answer{i}'] = ans

                await save_result_to_file(output_file, result)
                pbar.update(1)
                return result

            except asyncio.TimeoutError:
                tqdm.write(f"[{idx}/{total}] ⏱️  Timeout (attempt {attempt}/{config['max_retries']})")
                if attempt < config['max_retries']:
                    await asyncio.sleep(2)
                    continue
            except json.JSONDecodeError as e:
                tqdm.write(f"[{idx}/{total}] ❌ JSON error (attempt {attempt}): {str(e)[:60]}")
                if attempt < config['max_retries']:
                    await asyncio.sleep(1)
                    continue
            except Exception as e:
                tqdm.write(f"[{idx}/{total}] ❌ Error (attempt {attempt}): {str(e)[:60]}")
                if attempt < config['max_retries']:
                    await asyncio.sleep(1)
                    continue

        # All retries failed
        tqdm.write(f"[{idx}/{total}] 🚨 Failed after {config['max_retries']} attempts: {title[:50]}")
        error_result = {
            "link": link,
            "title": title,
            "date": date,
            "section": config['section'],
            config['section']: item.get(config['section'], ''),
            "cloze_question": "ERROR",
            "num_blanks": config['num_blanks'],
            "error": "All retries failed"
        }
        
        # Add error answers
        for i in range(1, config['num_blanks'] + 1):
            error_result[f'answer{i}'] = "ERROR"

        await save_result_to_file(output_file, error_result)
        pbar.update(1)
        return error_result


async def process_items(input_file: str, output_file: str, config: Dict):
    """Process all items and generate CLOZE questions"""

    print("Loading data...")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    print(f"Total records: {total}")
    print(f"Section: {config['section']}")
    print(f"Blanks per question: {config['num_blanks']}")
    print(f"Answer word limit: {config['min_words']}-{config['max_words']} words")
    print(f"Processing: ALL {total} records\n")

    # Initialize output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump([], f)

    print(f"Output: {output_file}\n")

    # Create client
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    semaphore = asyncio.Semaphore(config['concurrency'])

    # Create progress bar
    with tqdm(total=total, desc="Generating CLOZE questions", unit="question", colour="green") as pbar:
        # Create tasks
        tasks = [
            generate_single_cloze(client, item, idx, total, semaphore, output_file, pbar, config)
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


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Generate CLOZE questions from research papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate from abstracts (default)
  python generateCLOZE.py -i papers.json -o cloze_abstracts.json
  
  # Generate from introduction with 7 blanks
  python generateCLOZE.py -i papers.json -o cloze_intro.json -s introduction -n 7
  
  # Customize word limits
  python generateCLOZE.py -i papers.json -o cloze.json --min-words 2 --max-words 10
  
  # Use custom prompts
  python generateCLOZE.py -i papers.json -o cloze.json --system-prompt "Custom system prompt" --user-prompt "Custom {section} prompt"
        """
    )
    
    # Required arguments
    parser.add_argument('-i', '--input', required=True,
                        help='Input JSON file with paper data')
    parser.add_argument('-o', '--output', required=True,
                        help='Output JSON file for CLOZE questions')
    
    # Section and blanks configuration
    parser.add_argument('-s', '--section', default='abstract',
                        help='Section to use for CLOZE generation (default: abstract)')
    parser.add_argument('-n', '--num-blanks', type=int, default=DEFAULT_NUM_BLANKS,
                        help=f'Number of blanks per question (default: {DEFAULT_NUM_BLANKS})')
    
    # Answer constraints
    parser.add_argument('--min-words', type=int, default=DEFAULT_MIN_WORDS,
                        help=f'Minimum words per answer (default: {DEFAULT_MIN_WORDS})')
    parser.add_argument('--max-words', type=int, default=DEFAULT_MAX_WORDS,
                        help=f'Maximum words per answer (default: {DEFAULT_MAX_WORDS})')
    
    # API configuration
    parser.add_argument('-m', '--model', default='gpt-4o-mini',
                        help='OpenAI model to use (default: gpt-4o-mini)')
    parser.add_argument('-c', '--concurrency', type=int, default=DEFAULT_CONCURRENCY,
                        help=f'Number of concurrent API calls (default: {DEFAULT_CONCURRENCY})')
    parser.add_argument('-t', '--timeout', type=int, default=DEFAULT_TIMEOUT,
                        help=f'API timeout in seconds (default: {DEFAULT_TIMEOUT})')
    parser.add_argument('-r', '--max-retries', type=int, default=DEFAULT_MAX_RETRIES,
                        help=f'Maximum retries per question (default: {DEFAULT_MAX_RETRIES})')
    
    # Custom prompts
    parser.add_argument('--system-prompt', 
                        help='Custom system prompt (overrides default)')
    parser.add_argument('--user-prompt',
                        help='Custom user prompt template. Available variables: {link}, {title}, {date}, {section}, {content}, {num_blanks}, {min_words}, {max_words}')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.input).exists():
        print(f"❌ Error: Input file '{args.input}' not found")
        return
    
    if args.num_blanks < 1 or args.num_blanks > 20:
        print(f"❌ Error: Number of blanks must be between 1 and 20")
        return
    
    if args.min_words < 1 or args.max_words < args.min_words:
        print(f"❌ Error: Invalid word limits (min: {args.min_words}, max: {args.max_words})")
        return
    
    # Build configuration
    config = {
        'section': args.section,
        'num_blanks': args.num_blanks,
        'min_words': args.min_words,
        'max_words': args.max_words,
        'model': args.model,
        'concurrency': args.concurrency,
        'timeout': args.timeout,
        'max_retries': args.max_retries,
        'custom_system_prompt': args.system_prompt,
        'custom_user_prompt': args.user_prompt
    }
    
    # Print configuration
    print("=" * 70)
    print("CLOZE GENERATION CONFIGURATION")
    print("=" * 70)
    print(f"Input file:       {args.input}")
    print(f"Output file:      {args.output}")
    print(f"Section:          {args.section}")
    print(f"Blanks per Q:     {args.num_blanks}")
    print(f"Answer words:     {args.min_words}-{args.max_words}")
    print(f"Model:            {args.model}")
    print(f"Concurrency:      {args.concurrency}")
    print(f"Timeout:          {args.timeout}s")
    print(f"Max retries:      {args.max_retries}")
    if args.system_prompt:
        print(f"Custom system:    Yes")
    if args.user_prompt:
        print(f"Custom user:      Yes")
    print("=" * 70)
    print()
    
    # Run generation
    results = asyncio.run(process_items(args.input, args.output, config))
    
    success = sum(1 for r in results if r.get('error') is None)
    failed = len(results) - success
    
    print("\n" + "=" * 70)
    print(f"✅ Generation Success: {success}/{len(results)}")
    if failed > 0:
        print(f"❌ Generation Failed: {failed}/{len(results)}")
    print(f"💾 Output: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
