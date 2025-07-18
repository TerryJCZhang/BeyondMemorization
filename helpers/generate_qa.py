#!/usr/bin/env python3
"""
Question-Answer Generator: Generate High-Quality QA Pairs from Theorems

This script processes a dataset of theorems to generate high-quality question-answer pairs.
It works by:
1. Loading a dataset of scientifically significant theorems
2. For each theorem, using an LLM to generate a question-answer pair
3. Filtering based on quality criteria and saving to a new dataset

Key features:
- Uses o4-mini or custom LLM to generate QA pairs
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
import tempfile, shutil
import subprocess
from datasets import Dataset, load_from_disk, concatenate_datasets
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



class QAGenerator:
    """
    A class for generating high-quality question-answer pairs from theorems.
    
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
                    model="o4-mini-2025-04-16", 
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
                    break
                if iteration > 5:
                    console.print(f"[bold red]Failed to return good QA pair after {iteration} attempts[/bold red]")
                    return default_result
            
            return result
            
        except Exception as e:
            console.print(f"[bold red]Error generating QA pair: {e}[/bold red]")
            return default_result
    
    def process_dataset(self, input_dataset, output_path, sample_theorems=None):
        """
        Process a dataset of theorems to generate QA pairs.
        
        Args:
            input_dataset (Dataset): Dataset containing theorems
            output_path (str): Path to save the output QA dataset
            sample_theorems (int, optional): Number of theorems to process
            
        Returns:
            Dataset: Dataset of QA pairs
        """
        # Shuffle the dataset
        input_dataset = input_dataset.shuffle(seed=42)
        console.print(f"[green]Loaded {len(input_dataset)} theorems[/green]")
        
        # Sample if requested
        if sample_theorems:
            input_dataset = input_dataset.select(range(min(sample_theorems, len(input_dataset))))
            console.print(f"[yellow]Sampled {len(input_dataset)} theorems for processing[/yellow]")
        
        # Initialize containers for output dataset
        paper_links = []
        paper_ids = []
        paper_domains = []
        paper_citations = []
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
                paper_ids.append(entry['paper_id'])
                paper_domains.append(entry['paper_domain'])
                paper_citations.append(entry['paper_citations'])
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
            'paper_id': paper_ids,
            'paper_domain': paper_domains,
            'paper_citations': paper_citations,
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
    filtered_paper_ids = []
    filtered_paper_domains = []
    filtered_paper_citations = []
    filtered_theorems = []
    filtered_questions = []
    filtered_answers = []
    filtered_contexts = []
    
    # Track statistics
    total_samples = len(dataset)
    non_trivial_count = 0
    
    # System prompt for detecting trivial samples
    system_prompt = """You are an expert professor in charge of evaluating the quality of scientific question-answer pairs.
    
    Your job is to determine if a question-answer pair is "trivial" based on:
    1. Whether the answer can be directly found in the context or question
    2. Whether the answer can be easily guessed without deep understanding of relevant domain knowledge
    3. Whether the answer follows trivially from the question with 1-3 formulas required
    
    A good, non-trivial question requires deep understanding of domain knowledge and relevant theory, not just information retrieval.
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
        user_prompt = f"""Please evaluate if the following scientific question-answer pair is trivial:
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:
        {answer}
        
        A "trivial" question-answer pair means:
        1. The answer can be directly spotted in the context or question text
        2. The answer is too obvious and can be guessed without domain knowledge
        3. The question requires minimal scientific insight to solve
        
        Return your evaluation strictly in the following JSON format:
        {{
            "explanation": "Detailed explanation of why this sample is trivial or non-trivial",
            "is_trivial": "true" if the sample is trivial, "false" if it requires deep scientific understanding of relevant theory
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
                    model="gpt-4.1-2025-04-14",
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
            filtered_paper_ids.append(entry['paper_id'])
            filtered_paper_domains.append(entry['paper_domain'])
            filtered_paper_citations.append(entry['paper_citations'])
            filtered_theorems.append(entry['theorem'])
            filtered_questions.append(question)
            filtered_answers.append(answer)
            filtered_contexts.append(context)
            non_trivial_count += 1
    
    # Create the filtered dataset
    filtered_dataset = Dataset.from_dict({
        'paper_link': filtered_paper_links,
        'paper_id': filtered_paper_ids,
        'paper_domain': filtered_paper_domains,
        'paper_citations': filtered_paper_citations,
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

def find_all_theorems_dirs(root_dir):
    """Find all directories named "theorems" in the given root directory."""
    theorems_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if os.path.basename(dirpath) == "theorems":
            theorems_dirs.append(dirpath)
    return theorems_dirs


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Generate high-quality QA pairs from theorems")
    parser.add_argument("--input", type=str, default="output", help="Root directory containing nested theorems folders")
    parser.add_argument("--output", type=str, default="output", help="Root directory to save nested QA folders")
    parser.add_argument("--sample_theorems", type=int, help="Number of theorems to process")
    parser.add_argument("--filter_trivial", type=bool, default=True, help="Filter out trivial QA pairs")
    parser.add_argument("--append", action="store_true", help="Append to existing QA dataset instead of overwriting")
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    setup_random_seed(seed=42)
    
    console.print(
        Panel(
            "This tool generates high-quality question-answer pairs from scientific theorems.\n"
            "The QA pairs are filtered for quality, uniqueness, and proper formatting.",
            title="QA Pair Generator",
            border_style="blue"
        )
    )

    # Find all theorems folders
    theorems_dirs = find_all_theorems_dirs(args.input)
    if not theorems_dirs:
        console.print(f"[red]No theorems folders found in {args.input}[/red]")
        return
    
    # Create an instance of QAGenerator
    generator = QAGenerator()
    
    for theorems_dir in theorems_dirs:
        rel_path = os.path.relpath(theorems_dir, args.input)
        output_path = os.path.join(args.output, os.path.dirname(rel_path), "qa_pairs")
        os.makedirs(output_path, exist_ok=True)

        # APPEND MODE: Load already processed paper_ids
        already_processed_ids = set()
        if args.append and os.path.exists(output_path):
            try:
                processed_file = os.path.join(output_path, "processed_ids.json")
                if os.path.exists(processed_file):
                    with open(processed_file, "r") as f:
                        already_processed_ids = set(json.load(f))
                console.print(f"[yellow]Found {len(already_processed_ids)} already processed QAs in {output_path}[/yellow]")
            except Exception as e:
                console.print(f"[red]Could not load existing QA dataset: {e}[/red]")

        # Filter input dataset to only new theorems
        input_dataset = load_from_disk(theorems_dir)
        if args.append and already_processed_ids:
            input_dataset = input_dataset.filter(lambda theorem: theorem['paper_id'] not in already_processed_ids)
            if len(input_dataset) == 0:
                console.print(f"[green]No new theorems to process in {theorems_dir}[/green]")
                continue

        console.print(f"[bold]Processing theorems dataset: {theorems_dir}[/bold]")
        dataset = generator.process_dataset(
            input_dataset=input_dataset,
            output_path=output_path,
            sample_theorems=args.sample_theorems
        )
        
        # Filter trivial samples if requested
        if args.filter_trivial:
            console.print("[yellow]Filtering trivial samples from the dataset...[/yellow]")
            dataset = filter_trivial_samples(dataset)

        # Append to existing dataset if needed
        if args.append:
            try:
                if os.path.exists(os.path.join(output_path, "dataset_info.json")):
                    existing_dataset = load_from_disk(output_path)
                    dataset = concatenate_datasets([existing_dataset, dataset])
                else:
                    console.print(f"[yellow]No existing QA dataset found at {output_path}, skipping append.[/yellow]")
            except Exception as e:
                console.print(f"[red]Could not append to existing QA dataset: {e}[/red]")

        # Save to a temporary directory first to avoid overwrite error
        with tempfile.TemporaryDirectory() as tmp_save_path:
            dataset.save_to_disk(tmp_save_path)
            shutil.rmtree(output_path)
            shutil.move(tmp_save_path, output_path)

        console.print(f"[green]Filtered QA pairs dataset saved to {output_path}[/green]")

        # Save processed IDs (if in append mode)
        if args.append and os.path.exists(output_path):
            for theorem in input_dataset:
                already_processed_ids.add(theorem['paper_id'])
            with open(os.path.join(output_path, "processed_ids.json"), "w") as f:
                json.dump(list(already_processed_ids), f)
            console.print(f"[bold]Updated processed_ids.json with {len(already_processed_ids)} newly processed IDs.[/bold]")

        # Load and show final statistics
        if os.path.exists(output_path):
            final_dataset = Dataset.load_from_disk(output_path)
            console.print(f"[green]Current extract output contains {len(final_dataset)} QA pairs.[/green]")
        else:
            console.warning("[red]No output dataset found. No QA pairs may have been successfully processed.[/red]")

if __name__ == "__main__":
    main() 
