#!/usr/bin/env python3
"""
Question-Answer Generator: Generate High-Quality QA Pairs from Mathematics Theorems

This script processes a dataset of mathematical theorems to generate high-quality question-answer pairs.
It works by:
1. Loading a dataset of mathematically significant theorems
2. For each theorem, using an LLM to generate a question-answer pair
3. Filtering based on quality criteria and saving to a new dataset

Key features:
- Uses GPT-4o or custom LLM to generate QA pairs
- Enforces strict criteria for question-answer quality
- Outputs a structured dataset of QA pairs with links to original theorems

Usage:
  python generate_qa.py --input <theorem_dataset_path> [--output <qa_dataset_path>] [--sample_theorems <num>]
  
  For dataset processing:
    python generate_qa.py --input theorem_dataset --output qa_pairs

Dependencies:
  - OpenAI API access (for LLM-based QA generation)
  - tqdm, datasets, rich (for processing and output)
"""

import os
import argparse
import json
import tempfile
import subprocess
from datasets import Dataset, load_from_disk
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from prompts import SYSTEM_PROMPT_GENERATE_QA_FROM_THEOREMS_DATASET

# Load environment variables and set up console
load_dotenv()
console = Console()

# Default API key from environment variable
DEFAULT_API_KEY = os.getenv("OPENAI_API_KEY")

def setup_random_seed(seed=42):
    """
    Set the random seed for reproducibility.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

def validate_latex(content):
    """
    Validate if LaTeX content can be rendered correctly.
    
    Args:
        content (str): LaTeX content to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Create a temporary LaTeX document to test the content
    latex_document = f"""\\documentclass{{article}}
    \\usepackage{{amsmath, amssymb, enumerate, amsfonts, mathrsfs, mathtools, logicproof}}
    \\usepackage{{geometry}}
    \\usepackage{{hyperref}}
    \\usepackage{{xcolor}}
    \\usepackage{{fancyhdr}}
    \\usepackage{{tcolorbox}}
    \\begin{{document}}
    
    {content}
    
    \\end{{document}}
    """
    
    # Try to compile the LaTeX document
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create the LaTeX file
        tex_file_path = os.path.join(temp_dir, "output.tex")
        with open(tex_file_path, "w", encoding="utf-8") as f:
            f.write(latex_document)
        
        # Try to compile the LaTeX file
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_file_path],
                capture_output=True,
                cwd=temp_dir
            )
            
            if result.returncode != 0:
                # If compilation failed, return the error message
                return False, result.stderr.decode('utf-8', errors='replace')
            
            return True, ""
            
        except Exception as e:
            return False, str(e)

def fix_latex_formatting(client, content, error_message=""):
    """
    Fix LaTeX formatting issues.
    
    Args:
        client: OpenAI client
        content (str): LaTeX content to fix
        error_message (str): Error message from LaTeX compilation
        
    Returns:
        str: Fixed LaTeX content
    """
    system_prompt = r"""You are an expert in LaTeX formatting. Your task is to fix formatting issues in LaTeX content 
    so it can be properly rendered. Focus only on fixing LaTeX syntax errors, not content.
    
    Common LaTeX formatting issues to fix:
    1. Unescaped special characters (e.g., $ in text, % in text)
    2. Missing or mismatched delimiters ($, \[, \], etc.)
    3. Undefined commands or environments
    4. Improper nesting of environments
    5. Malformed mathematical expressions
    
    Only use standard LaTeX packages: amsmath, amssymb, enumerate, amsfonts, mathrsfs, mathtools, logicproof.
    Make sure all mathematical formulas are enclosed in $ or $$ or \[ \] properly.
    
    Return the fixed content.
    """
    
    user_prompt = f"""Please fix the LaTeX formatting issues in the following content:
    
    CONTENT:
    {content}
    
    ERROR MESSAGE (if any):
    {error_message}
    
    Return only the fixed content without any explanations.
    """
    
    try:
        response = client.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        fixed_content = response.choices[0].message.content.strip()
        return fixed_content
        
    except Exception as e:
        console.print(f"[bold red]Error fixing LaTeX: {e}[/bold red]")
        return content

class QAGenerator:
    """
    A class for generating high-quality question-answer pairs from mathematical theorems.
    
    This class provides functionality to:
    1. Process theorem datasets to generate QA pairs
    2. Apply quality filters to ensure high-quality outputs
    3. Save the resulting QA pairs to a new dataset
    """
    
    def __init__(self, api_key=DEFAULT_API_KEY):
        """
        Initialize the QAGenerator.
        
        Args:
            api_key (str): API key for LLM access
        """
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        
    def generate_qa_pair(self, theorem):
        """
        Generate a question-answer pair from a theorem.
        
        Args:
            theorem (str): The theorem to generate a QA pair from
            
        Returns:
            dict: A dictionary containing question, answer, and whether the theorem is good
        """
        default_result = {
            "question": "",
            "answer": "",
            "is_good_qa": "false"
        }
        
        try:
            # Updated user prompt with LaTeX formatting instructions
            user_prompt = f"""Please create a question-answer pair from this theorem:
            
            THEOREM:
            {theorem}
            
            Generate a question-answer pair based on this theorem.
            
            IMPORTANT: Make sure both the question and answer are formatted with proper LaTeX syntax:
            1. All mathematical expressions must be enclosed in $ for inline math or $$ or \\[ \\] for display math
            2. Use standard LaTeX commands for mathematical symbols (\\alpha, \\beta, \\sum, etc.)
            3. Format the output so it can be directly rendered in a LaTeX document
            
            Don't mention the answer in the question, it's a bad practice.
            Return your output strictly in the following JSON format:
            {{
                "question": "Clearly stated, unique-answer question derived from the theorem. if the theorem is not good, return an empty string",
                "answer": "The single, unique, exact answer derived from the theorem. if the theorem is not good, return an empty string",
                "is_good_qa": "true" if the question-answer pair is good, otherwise "false"
            }}
            """
            
            # Call the LLM to generate a QA pair with retries
            iteration = 0
            while True:
                iteration += 1
                response = self.client.chat.completions.create(
                    model="o3-mini-2025-01-31", 
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_GENERATE_QA_FROM_THEOREMS_DATASET},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                )
                
                # Parse the response to get the QA pair
                response_content = response.choices[0].message.content
                result = json.loads(response_content)
                
                # Validate the response
                if all(key in result for key in default_result.keys()) and result['is_good_qa'] == 'true':
                    # for quick testing, we skip LaTeX validation
                    validate_latex = False
                    if validate_latex:
                        # Validate and fix LaTeX formatting for question and answer
                        question = result['question']
                        answer = result['answer']
                        # Validate question LaTeX
                        question_valid, question_error = validate_latex(question)
                        if not question_valid:
                            console.print(f"[yellow]Question LaTeX has formatting issues. Attempting to fix...[/yellow]")
                            question = fix_latex_formatting(self.client, question, question_error)
                            question_valid, _ = validate_latex(question)
                    
                        # Validate answer LaTeX
                        answer_valid, answer_error = validate_latex(answer)
                        if not answer_valid:
                            console.print(f"[yellow]Answer LaTeX has formatting issues. Attempting to fix...[/yellow]")
                            answer = fix_latex_formatting(self.client, answer, answer_error)
                            answer_valid, _ = validate_latex(answer)
                        
                        # Only accept if both question and answer have valid LaTeX
                        if question_valid and answer_valid:
                            result['question'] = question
                            result['answer'] = answer
                            break
                        elif iteration > 3:
                            # After 3 attempts, just continue with what we have
                            result['question'] = question
                            result['answer'] = answer
                            console.print(f"[yellow]Warning: After {iteration} attempts, LaTeX validation still shows issues. Proceeding anyway.[/yellow]")
                            break
                    else:
                        break
                if iteration > 5:
                    console.print(f"[bold red]Failed to return good QA pair after {iteration} attempts[/bold red]")
                    return default_result
            
            return result
            
        except Exception as e:
            console.print(f"[bold red]Error generating QA pair: {e}[/bold red]")
            return default_result
    
    def process_dataset(self, input_path, output_path, sample_theorems=None):
        """
        Process a dataset of theorems to generate QA pairs.
        
        Args:
            input_path (str): Path to the input theorem dataset
            output_path (str): Path to save the output QA dataset
            sample_theorems (int, optional): Number of theorems to process
            
        Returns:
            Dataset: Dataset of QA pairs
        """
        # Load the input dataset
        console.print(f"[green]Loading theorem dataset from {input_path}...[/green]")
        input_dataset = load_from_disk(input_path)
        
        # Shuffle the dataset
        input_dataset = input_dataset.shuffle(seed=42)
        console.print(f"[green]Loaded {len(input_dataset)} theorems[/green]")
        
        # Sample if requested
        if sample_theorems:
            input_dataset = input_dataset.select(range(min(sample_theorems, len(input_dataset))))
            console.print(f"[yellow]Sampled {len(input_dataset)} theorems for processing[/yellow]")
        
        # Initialize containers for output dataset
        paper_links = []
        theorems = []
        questions = []
        answers = []
        contexts = []  # Add a new list for contexts
        
        # Process each theorem
        total_theorems = len(input_dataset)
        good_theorems = 0
        
        for i, entry in enumerate(tqdm(input_dataset, total=total_theorems, desc="Generating QA pairs")):
            console.print(f"[bold]Processing theorem {i+1}/{total_theorems}[/bold]")
            
            # Extract theorem
            theorem = entry['theorem']
            paper_link = entry['paper_link']
            context = entry['context']
            
            # Generate QA pair
            qa_pair = self.generate_qa_pair(theorem)
            
            if qa_pair['question'] and qa_pair['answer']:
                # Add to output
                paper_links.append(paper_link)
                theorems.append(theorem)
                questions.append(qa_pair['question'])
                answers.append(qa_pair['answer'])
                contexts.append(context)  # Add context to output
                good_theorems += 1
                
                console.print(f"[green]Generated QA pair {good_theorems}[/green]")
                console.print(f"[bold]Question:[/bold] {qa_pair['question'][:100]}...")
                console.print(f"[bold]Answer:[/bold] {qa_pair['answer'][:100]}...")
            else:
                console.print(f"[yellow]Skipping theorem {i+1} - not suitable for QA pair[/yellow]")
        
        # Create the output dataset
        output_dataset = Dataset.from_dict({
            'paper_link': paper_links,
            'theorem': theorems,
            'question': questions,
            'answer': answers,
            'context': contexts  
        })
        
        # Print statistics
        console.print(
            Panel(
                f"[bold green]Processing complete![/bold green]\n\n"
                f"Total theorems processed: {total_theorems}\n"
                f"Good theorems with QA pairs: {good_theorems}\n"
                f"Conversion rate: {good_theorems / total_theorems * 100:.1f}%",
                title="QA Generation Results",
                border_style="green"
            )
        )
        
        # Save the output dataset
        output_dataset.save_to_disk(output_path)
        console.print(f"[green]Saved QA dataset to {output_path}[/green]")
        
        return output_dataset


def filter_trivial_samples(dataset):
    """
    Filter out trivial samples from the dataset.
    
    A sample is considered trivial if:
    1. The answer can be directly found in the context or question
    2. The answer can be easily guessed without understanding the theorem
    
    Args:
        dataset (Dataset): The dataset of QA pairs to filter
        
    Returns:
        Dataset: Filtered dataset with non-trivial QA pairs
    """
    console.print(
        Panel(
            "Filtering trivial QA pairs from the dataset...\n"
            "A sample is considered trivial if the answer can be easily found or guessed.",
            title="Trivial Sample Filter",
            border_style="yellow"
        )
    )
    
    # Initialize OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.OpenAI(api_key=api_key)
    
    # Initialize containers for filtered dataset
    filtered_paper_links = []
    filtered_theorems = []
    filtered_questions = []
    filtered_answers = []
    filtered_contexts = []
    
    # Track statistics
    total_samples = len(dataset)
    non_trivial_count = 0
    
    # System prompt for detecting trivial samples
    system_prompt = """You are an expert mathematics educator tasked with evaluating the quality of mathematics question-answer pairs.
    
    Your job is to determine if a question-answer pair is "trivial" based on:
    1. Whether the answer can be directly found in the context or question
    2. Whether the answer can be easily guessed without deep understanding of mathematics
    3. Whether the answer follows trivially from the question with minimal mathematical insight required
    
    A good, non-trivial question requires mathematical understanding, not just information retrieval.
    """
    
    # Default result in case of persistent failures
    default_result = {
        "explanation": "Failed to evaluate due to API errors",
        "is_trivial": "false"  # Be conservative and keep samples when in doubt
    }
    
    # Process each sample
    for i, entry in enumerate(tqdm(dataset, total=total_samples, desc="Evaluating samples")):
        context = entry['context']
        question = entry['question']
        answer = entry['answer']
        
        # Construct the user prompt
        user_prompt = f"""Please evaluate if the following mathematics question-answer pair is trivial:
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:
        {answer}
        
        A "trivial" question-answer pair means:
        1. The answer can be directly spotted in the context or question text
        2. The answer is too obvious and can be guessed without mathematical knowledge
        3. The question requires minimal mathematical insight to solve
        
        Return your evaluation strictly in the following JSON format:
        {{
            "explanation": "Detailed explanation of why this sample is trivial or non-trivial",
            "is_trivial": "true" if the sample is trivial, "false" if it requires mathematical understanding
        }}
        """
        
        # Initialize result
        result = default_result
        
        # Try calling the API with retries
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                # Call the LLM to evaluate the sample
                response = client.chat.completions.create(
                    model="o3-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                )
                
                # Parse the response
                response_content = response.choices[0].message.content
                result = json.loads(response_content)
                
                # If we got a valid response, break out of the retry loop
                if all(key in result for key in ["explanation", "is_trivial"]):
                    break
                    
            except Exception as e:
                # Log the error
                if attempt < max_retries:
                    console.print(f"[yellow]Attempt {attempt}/{max_retries} failed: {e}. Retrying...[/yellow]")
                else:
                    console.print(f"[bold red]All {max_retries} attempts failed for sample {i+1}: {e}[/bold red]")
                    # Use default result after all retries fail
                    result = default_result
            
        # Display the evaluation result
        console.print(f"[bold]Sample {i+1}/{total_samples}[/bold]")
        console.print(f"[bold]Question:[/bold] {question[:100]}...")
        is_trivial = result.get('is_trivial', 'false') == 'true'
        
        if is_trivial:
            console.print(f"[red]Trivial sample detected.[/red]")
            console.print(f"[bold]Explanation:[/bold] {result.get('explanation', 'No explanation provided')}")
        else:
            console.print(f"[green]Non-trivial sample identified.[/green]")
            # Add to filtered output
            filtered_paper_links.append(entry['paper_link'])
            filtered_theorems.append(entry['theorem'])
            filtered_questions.append(question)
            filtered_answers.append(answer)
            filtered_contexts.append(context)
            non_trivial_count += 1
    
    # Create the filtered dataset
    filtered_dataset = Dataset.from_dict({
        'paper_link': filtered_paper_links,
        'theorem': filtered_theorems,
        'question': filtered_questions,
        'answer': filtered_answers,
        'context': filtered_contexts
    })
    
    # Print statistics
    console.print(
        Panel(
            f"[bold green]Filtering complete![/bold green]\n\n"
            f"Total samples evaluated: {total_samples}\n"
            f"Non-trivial samples retained: {non_trivial_count}\n"
            f"Trivial samples removed: {total_samples - non_trivial_count}\n"
            f"Retention rate: {non_trivial_count / total_samples * 100:.1f}%",
            title="Filtering Results",
            border_style="green"
        )
    )
    
    return filtered_dataset

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate high-quality QA pairs from mathematics theorems")
    parser.add_argument("--input", type=str, default="theorem_dataset", help="Path to the input theorem dataset")
    parser.add_argument("--output", type=str, default="qa_pairs", help="Path to save the output QA dataset")
    parser.add_argument("--sample_theorems", type=int, help="Number of theorems to process")
    parser.add_argument("--filter_trivial", type=bool, default=True, help="Filter out trivial QA pairs")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    setup_random_seed(seed=42)
    
    console.print(
        Panel(
            "This tool generates high-quality question-answer pairs from mathematical theorems.\n"
            "The QA pairs are filtered for quality, uniqueness, and proper formatting.",
            title="QA Pair Generator",
            border_style="blue"
        )
    )
    
    # Create an instance of QAGenerator
    generator = QAGenerator()
    
    # Process the theorem dataset
    dataset = generator.process_dataset(
        input_path=args.input,
        output_path=args.output,
        sample_theorems=args.sample_theorems
    )
    import pdb; pdb.set_trace()
    # Filter trivial samples if requested
    if args.filter_trivial:
        # dataset = load_from_disk(args.input)
        console.print("[yellow]Filtering trivial samples from the dataset...[/yellow]")
        dataset = filter_trivial_samples(dataset)
        
        # Save the filtered dataset
        filtered_output_path = f"{args.output}_filtered"
        dataset.save_to_disk(filtered_output_path)
        console.print(f"[green]Filtered QA pairs dataset saved to {filtered_output_path}[/green]")
    
    console.print(f"[bold]QA pairs dataset saved to {args.output}[/bold]")

if __name__ == "__main__":
    main() 