"""
Math QA Evaluator: Evaluate LLM Performance on Mathematical Problem-Solving

This script evaluates the performance of Language Models (LLMs) on mathematical
question-answering tasks using a dataset generated from mathematical papers.

The script:
1. Loads a mathematical QA dataset (containing questions, context, and ground truth answers)
2. For each question, prompts an LLM to solve it using the provided context
3. Evaluates the correctness of the LLM's answer against the ground truth
4. Generates detailed metrics on the model's performance

The script supports accessing models through:
- OpenAI API (for GPT models)
- Anthropic API (for Claude models)
- OpenRouter API (for various models like DeepSeek-R1)

Usage:
  # Evaluate a model
  python math_qa_evaluator.py --dataset <path> --model <model_name> [--sample <n>] [--verbose]

Dependencies:
  - datasets (for loading the dataset)
  - openai (for OpenAI models, OpenRouter models, and evaluation)
  - anthropic (for Claude models)
  - numpy (for metrics calculation)

Environment Variables:
  - OPENAI_API_KEY: API key for OpenAI
  - ANTHROPIC_API_KEY: API key for Anthropic
  - OPENROUTER_API_KEY: API key for OpenRouter
"""

import argparse
import os
import json
import time
import random
from tqdm import tqdm
import numpy as np
from datasets import load_from_disk
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    console.print("[bold red]Hugging Face Datasets not installed properly. HF datasets will not be available.[/bold red]")
    console.print("[yellow]To enable HF datasets, install with: pip install datasets[/yellow]")
    HF_AVAILABLE = False
# use dotenv to load the api keys
from dotenv import load_dotenv
load_dotenv()
import re

# Import rich console for better formatting
from rich.console import Console
from rich.panel import Panel
console = Console()

# Handle optional dependencies with graceful fallbacks
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    console.print("[bold red]OpenAI SDK not found. OpenAI models will not be available.[/bold red]")
    console.print("[yellow]To enable OpenAI models, install with: pip install openai[/yellow]")
    OPENAI_AVAILABLE = False

# Add Anthropic support
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    console.print("[bold red]Anthropic SDK not found. Claude models will not be available.[/bold red]")
    console.print("[yellow]To enable Claude models, install with: pip install anthropic[/yellow]")
    ANTHROPIC_AVAILABLE = False

# Add OpenRouter support (already uses OpenAI client)
OPENROUTER_AVAILABLE = OPENAI_AVAILABLE  # Depends on OpenAI client

# Add wandb support
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    console.print("[bold red]Weights & Biases not installed. wandb logging will not be available.[/bold red]")
    console.print("[yellow]To enable wandb, install with: pip install wandb[/yellow]")
    WANDB_AVAILABLE = False


OPENAI_MODELS = {
    "o3-mini": "o3-mini-2025-01-31",
    "o1-mini": "o1-mini-2024-09-12",    # call error
    "gpt-3.5-turbo": "gpt-3.5-turbo-0125",  # context window too small
    "o4-mini": "o4-mini-2025-04-16", 
    "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
    "o3": "o3-2025-04-16",
    # "gpt-4o-mini-ft": "ft:gpt-4o-mini-2024-07-18:eth-zurich-florian-tramer-lab:math-arxiv:BU1QAVtm",
    "gpt-4o-mini-ft": "ft:gpt-4o-mini-2024-07-18:eth-zurich-florian-tramer-lab::BUkNhf0o"
}

ANTHROPIC_MODELS = {
    "claude-3.7-sonnet": "claude-3-7-sonnet-20250219",
    "claude-3.5-sonnet": "claude-3-5-sonnet-20240620",
    
}

# Add OpenRouter models
OPENROUTER_MODELS = {
    "deepseek-r1": "deepseek/deepseek-r1",  # high latency
    "grok-3":"x-ai/grok-3-beta",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    "llama-3.1-8b": "meta-llama/llama-3.1-8b-instruct",
    "gemini-2.5-flash": "google/gemini-2.5-flash-preview",
    "gemini-2.5-pro": "google/gemini-2.5-pro-preview-03-25",    # 2025-03-25
    "qwen-32b": "qwen/qwq-32b",
    "qwen3-235b": "qwen/qwen3-235b-a22b"
    
    
}


class MathQAEvaluator:
    """
    A class for evaluating LLM performance on mathematical question-answering tasks.
    
    This class provides functionality to:
    1. Load a mathematical QA dataset
    2. Query OpenAI GPT models with mathematical problems
    3. Evaluate the correctness of the LLM's answers
    4. Generate performance metrics and comparisons
    """
    
    def __init__(self, verbose=False):
        """
        Initialize the MathQAEvaluator.
        
        Args:
            verbose (bool, optional): Whether to print detailed information. Defaults to False.
        """
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")
        self.verbose = verbose
        
        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None
        self.openrouter_client = None
        if OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
        if ANTHROPIC_AVAILABLE:
            self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
        if OPENROUTER_AVAILABLE and self.openrouter_api_key:
            self.openrouter_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key
            )
    
    def load_dataset(self, dataset_path, sample_size=None, subset=None):
        """
        Load the mathematical QA dataset.
        
        Args:
            dataset_path (str): Path to the dataset or Hugging Face dataset ID.
            sample_size (int, optional): Number of samples to use. Defaults to None (use entire dataset).
                                        If set to 0, will use the entire dataset.
            
        Returns:
            dataset: The loaded dataset.
        """
        dataset = None
        # First try to load from Hugging Face
        if HF_AVAILABLE and (dataset_path.startswith("ethz-spylab/") or "/" in dataset_path):
            try:
                if self.verbose:
                    console.print(f"[yellow]Attempting to load dataset from Hugging Face: {dataset_path}[/yellow]")
                if subset == "stackexchange":
                    dataset = load_dataset(dataset_path)
                else:
                    dataset = load_dataset(dataset_path, split=subset)
                # Convert to regular dataset (not DatasetDict)
                if isinstance(dataset, dict):
                    if "train" in dataset:
                        dataset = dataset["train"]
                    else:
                        # Use the first split
                        dataset = dataset[list(dataset.keys())[0]]
                console.print(f"[green]Successfully loaded dataset from Hugging Face: {dataset_path}[/green]")
            except Exception as e:
                console.print(f"[bold red]Error loading from Hugging Face: {e}[/bold red]")
                console.print("[yellow]Falling back to local dataset loading...[/yellow]")
                
        # If HF loading failed or not a HF path, try loading from disk
        if dataset is None:
            try:
                if self.verbose:
                    console.print(f"[yellow]Attempting to load dataset from disk: {dataset_path}[/yellow]")
                dataset = load_from_disk(dataset_path)
                console.print(f"[green]Successfully loaded dataset from disk: {dataset_path}[/green]")
            except Exception as e:
                console.print(f"[bold red]Error loading dataset from disk: {e}[/bold red]")
                return None
        
        # Handle sample size - use full dataset if sample_size is 0
        if sample_size and sample_size > 0 and sample_size < len(dataset):
            # Randomly sample from the dataset
            indices = random.sample(range(len(dataset)), sample_size)
            dataset = dataset.select(indices)
            if self.verbose:
                console.print(f"[yellow]Sampled {sample_size} examples from dataset with {len(dataset)} total examples[/yellow]")
                
        if self.verbose:
            console.print(f"[green]Final dataset has {len(dataset)} examples[/green]")
            
        return dataset
    
    def query_openai_model(self, context, question, model_name="gpt-4o", use_context=True):
        """
        Query an OpenAI model with a mathematical problem.
        
        Args:
            context (str): The context/background information for the problem.
            question (str): The mathematical question to solve.
            model_name (str, optional): The OpenAI model to use. Defaults to "gpt-4o".
            use_context (bool, optional): Whether to include context in the prompt. Defaults to True.
            
        Returns:
            str: The model's answer to the question.
        """
        default_answer = {"final_answer": "", "reasoning": ""}
            
        system_prompt = """You are an expert mathematician tasked with solving a mathematical problem. Given a question and context, provide a clear, step-by-step solution to the question based on the provided context.
        Your answer should be precise, rigorous, and use proper mathematical notation.
        
        After your detailed explanation, include your final answer in a clear, properly formatted LaTeX after a section titled 'Final Answer' (\\section*{{Final Answer}}).
        Ensure all math expressions are properly enclosed in $...$ or \\[...\\] delimiters.
        
        """
        
        if use_context:
            user_prompt = f"""CONTEXT:
            {context}
            
            QUESTION:
            {question}
            
            Please solve this mathematical problem step by step, showing your reasoning clearly.
            At the end, provide your final answer in well-formatted LaTeX under a section titled 'Final Answer' (\\section*{{Final Answer}})."""
        else:
            user_prompt = f"""QUESTION:
            {question}
            
            Please solve this mathematical problem step by step, showing your reasoning clearly.
            At the end, provide your final answer in well-formatted LaTeX under a section titled 'Final Answer' (\\section*{{Final Answer}})."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            )
            
            return response.choices[0].message.content
        except Exception as e:
            if self.verbose:
                console.print(f"[bold red]Error querying OpenAI model: {e}[/bold red]")
            return f"Error: {str(e)}"
    
    def query_antropic_model(self, context, question, model_name="claude-3-7-sonnet", use_context=True):
        """
        Query an Anthropic Claude model with a mathematical problem.
        
        Args:
            context (str): The context/background information for the problem.
            question (str): The mathematical question to solve.
            model_name (str, optional): The Anthropic model to use. Defaults to "claude-3-7-sonnet".
            use_context (bool, optional): Whether to include context in the prompt. Defaults to True.
            
        Returns:
            str: The model's answer to the question.
        """
        if not ANTHROPIC_AVAILABLE or not self.anthropic_client:
            return "Error: Anthropic client not available"
            
        system_prompt = """You are an expert mathematician tasked with solving a mathematical problem.
        Provide a clear, step-by-step solution to the question based on the provided context.
        Your answer should be precise, rigorous, and use proper mathematical notation.
        
        After your detailed explanation, include your final answer in a clear, properly formatted LaTeX form
        under a section titled 'Final Answer' like this:
        
        \\section*{Final Answer}
        Your well-formatted LaTeX answer here, with appropriate line breaks for complex expressions.
        Ensure all math expressions are properly enclosed in $...$ or \\[...\\] delimiters.
        
        Ensure that the LaTeX is valid and can be directly rendered in standard LaTeX without requiring custom command definitions.
        """
        
        if use_context:
            user_content = f"""CONTEXT:
            {context}
            
            QUESTION:
            {question}
            
            Please solve this mathematical problem step by step, showing your reasoning clearly.
            At the end, provide your final answer in well-formatted LaTeX under a section titled 'Final Answer' (\\section*{{Final Answer}})."""
        else:
            user_content = f"""QUESTION:
            {question}
            
            Please solve this mathematical problem step by step, showing your reasoning clearly.
            At the end, provide your final answer in well-formatted LaTeX under a section titled 'Final Answer' (\\section*{{Final Answer}})."""
        
        try:
            response = self.anthropic_client.messages.create(
                model=model_name,
                max_tokens=4000,
                messages=[
                    {"role": "user", "content": user_content}
                ],
                system=system_prompt
            )
            
            return response.content[0].text
        except Exception as e:
            if self.verbose:
                console.print(f"[bold red]Error querying Anthropic model: {e}[/bold red]")
            return f"Error: {str(e)}"
    
    def query_openrouter_models(self, context, question, model_name="deepseek/deepseek-r1", use_context=True):
        """
        Query an OpenRouter model with a mathematical problem.
        
        Args:
            context (str): The context/background information for the problem.
            question (str): The mathematical question to solve.
            model_name (str, optional): The OpenRouter model to use. Defaults to "deepseek/deepseek-r1".
            use_context (bool, optional): Whether to include context in the prompt. Defaults to True.
            
        Returns:
            str: The model's answer to the question.
        """
        if not OPENROUTER_AVAILABLE or not self.openrouter_client:
            return "Error: OpenRouter client not available"
            
        system_prompt = """You are an expert mathematician tasked with solving a mathematical problem.
        Provide a clear, step-by-step solution to the question based on the provided context.
        Your answer should be precise, rigorous, and use proper mathematical notation.
        
        After your detailed explanation, include your final answer in a clear, properly formatted LaTeX form
        under a section titled 'Final Answer' like this:
        
        \\section*{Final Answer}
        Your well-formatted LaTeX answer here, with appropriate line breaks for complex expressions.
        Ensure all math expressions are properly enclosed in $...$ or \\[...\\] delimiters.
        
        Ensure that the LaTeX is valid and can be directly rendered in standard LaTeX without requiring custom command definitions.
        """
        
        if use_context:
            user_prompt = f"""CONTEXT:
            {context}
            
            QUESTION:
            {question}
            
            Please solve this mathematical problem step by step, showing your reasoning clearly.
            At the end, provide your final answer in well-formatted LaTeX under a section titled 'Final Answer' (\\section*{{Final Answer}})."""
        else:
            user_prompt = f"""QUESTION:
            {question}
            
            Please solve this mathematical problem step by step, showing your reasoning clearly.
            At the end, provide your final answer in well-formatted LaTeX under a section titled 'Final Answer' (\\section*{{Final Answer}})."""
        
        try:
            response = self.openrouter_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                # stream=True,
                # extra_headers={
                #     "HTTP-Referer": "https://math-benchmark-eval.com",
                #     "X-Title": "Math Benchmark Evaluation"
                # }
            )
            
            return response.choices[0].message.content
        except Exception as e:
            if self.verbose:
                console.print(f"[bold red]Error querying OpenRouter model: {e}[/bold red]")
            return f"Error: {str(e)}"
    
    def query_model(self, context, question, model_name, use_context=True):
        """
        Query the appropriate model based on model name.
        
        Args:
            context (str): The context for the problem.
            question (str): The mathematical question.
            model_name (str): The name of the model to use.
            use_context (bool, optional): Whether to include context in the prompt. Defaults to True.
            
        Returns:
            str: The model's answer.
        """
        # OpenAI models
        if model_name in OPENAI_MODELS.keys():
            return self.query_openai_model(context, question, model_name=OPENAI_MODELS[model_name], use_context=use_context)
        # Anthropic models
        elif model_name in ANTHROPIC_MODELS.keys():
            return self.query_antropic_model(context, question, model_name=ANTHROPIC_MODELS[model_name], use_context=use_context)
        # OpenRouter models
        elif model_name in OPENROUTER_MODELS.keys():
            return self.query_openrouter_models(context, question, model_name=OPENROUTER_MODELS[model_name], use_context=use_context)
        # Unsupported model
        else:
            return f"Error: Unsupported model {model_name}. Only OpenAI GPT models, Anthropic Claude models, and OpenRouter models are supported."
    
    def verify_latex_compatibility(self, answer):
        """
        Verify that the LaTeX in the answer can be properly rendered
        in a standard LaTeX document environment with common math packages.
        
        Args:
            answer (str): LaTeX-formatted answer
            
        Returns:
            dict: Dictionary containing validated 'final_answer' and 'reasoning'
        """
        # Keep the full answer as the reasoning part
        reasoning = answer
        
        # Extract the final answer if it's in a Final Answer section
        final_answer = answer
        extracted_answer = None
        
        # Try to find the final answer section with different variations of the command
        section_patterns = [
            r'\\section\*{Final Answer}(.*?)(?:\\section|\Z)',  # \section*{Final Answer} until next section or end
            r'\\section{Final Answer}(.*?)(?:\\section|\Z)',    # \section{Final Answer} until next section or end
            r'\\subsection\*{Final Answer}(.*?)(?:\\section|\\subsection|\Z)', # subsection version
            r'\\subsection{Final Answer}(.*?)(?:\\section|\\subsection|\Z)',  # subsection version
            r'\[FINAL ANSWER\](.*?)\[/FINAL ANSWER\]'  # Legacy tag format
        ]
        
        # Try each pattern until we find a match
        for pattern in section_patterns:
            final_answer_match = re.search(pattern, answer, re.DOTALL)
            if final_answer_match:
                extracted_answer = final_answer_match.group(1).strip()
                if extracted_answer:
                    final_answer = extracted_answer
                    break
        
        # First, try direct compilation with pdflatex to check compatibility
        compile_success, fixed_answer = self.compile_test_latex(final_answer)
        
        # If compilation succeeded, use the validated content
        if compile_success:
            if self.verbose:
                console.print("LaTeX content successfully compiled with pdflatex")
            final_answer = fixed_answer
        else:
            # If compilation failed, try to fix the LaTeX with GPT-4o
            if not OPENAI_AVAILABLE or not self.openai_client:
                if self.verbose:
                    console.print("OpenAI client not available for LaTeX fixing")
            else:
                try:
                    system_prompt = """You are an expert in LaTeX. Your task is to review an answer from a mathematics problem and ensure it can be directly rendered in standard LaTeX without requiring custom command definitions.
                    
                    For any issues in the LaTeX:
                    1. Fix improper math environment delimiters (ensure all $ and \\[ \\] are properly paired)
                    2. Fix unmatched brackets, braces, or parentheses
                    3. Fix any command syntax errors
                    4. Replace any non-standard LaTeX commands with standard equivalents
                    5. Ensure all LaTeX is properly formatted with appropriate line breaks for complex expressions
                    6. Make sure mathematical formulas are properly enclosed in $ or \\[ \\] delimiters
                    
                    IMPORTANT: You must not change the mathematical meaning of the content. Focus only on syntax corrections.
                    
                    Respond ONLY with the corrected text. Do not explain your changes or add any comments."""
                    
                    user_prompt = f"""The following is an answer to a mathematics problem that may contain LaTeX errors or non-standard commands:

                    {final_answer}
                    
                    Please fix any LaTeX syntax issues to ensure it can be compiled in a standard LaTeX document with amsmath and amssymb packages.
                    Format the answer with proper line breaks for complex expressions and ensure all math is properly delimited.
                    Only make changes necessary for proper LaTeX rendering. Don't change the mathematical content or meaning."""
                    
                    response = self.openai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.0,
                        max_tokens=1000
                    )
                    
                    fixed_answer = response.choices[0].message.content.strip()
                    
                    if self.verbose and fixed_answer != final_answer:
                        console.print("LaTeX formatting fixed in the answer")
                        
                    final_answer = fixed_answer
                        
                except Exception as e:
                    if self.verbose:
                        console.print(f"[bold red]Error fixing LaTeX: {e}[/bold red]")
        
        return {"final_answer": final_answer, "reasoning": reasoning}
    
    def compile_test_latex(self, answer):
        """
        Test if the LaTeX content can be compiled using pdflatex.
        
        Args:
            answer (str): LaTeX-formatted answer
            
        Returns:
            tuple: (success_status, answer) - success_status is True if compilation succeeds
        """
        import subprocess
        import tempfile
        import os
        import shutil
        
        # First check if pdflatex is available
        pdflatex_available = shutil.which("pdflatex") is not None
        
        if not pdflatex_available:
            if self.verbose:
                console.print("pdflatex not found in PATH. Install LaTeX (TeX Live or MiKTeX) to enable direct compilation verification.")
            # Return failure but don't modify the content
            return False, answer
        
        # Create a comprehensive LaTeX document with all necessary math packages
        latex_document = r"""\documentclass{article}
                        \usepackage{amsmath}
                        \usepackage{amssymb}
                        \usepackage{amsthm}
                        \usepackage{mathtools}
                        \usepackage{bm}

                        \begin{document}

                        %s

                        \end{document}
                        """ % answer
        
        # Test compilation
        with tempfile.NamedTemporaryFile(suffix='.tex', delete=False) as f:
            tex_file = f.name
            f.write(latex_document.encode('utf-8'))
        
        try:
            result = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_file], 
                capture_output=True,
                cwd=os.path.dirname(tex_file)
            )
            success = result.returncode == 0
            
            # Clean up temporary files
            temp_dir = os.path.dirname(tex_file)
            base_name = os.path.splitext(os.path.basename(tex_file))[0]
            for ext in ['.tex', '.aux', '.log', '.pdf']:
                try:
                    os.remove(os.path.join(temp_dir, f"{base_name}{ext}"))
                except:
                    pass
                    
            return success, answer
        except Exception as e:
            if self.verbose:
                console.print(f"[bold red]Error during LaTeX compilation: {e}[/bold red]")
            return False, answer
    
    def evaluate_answer(self, answer_data, ground_truth, question):
        """
        Evaluate the correctness of the generated answer against the ground truth
        using GPT-4o as a judge.
        
        Args:
            answer_data (str): The generated answer.
            ground_truth (str): The ground truth answer.
            question (str): The original question.
            
        Returns:
            tuple: (bool, str) - Whether the answer is correct and an explanation
        """
        if not OPENAI_AVAILABLE or not self.openai_client:
            return False, "OpenAI client not available for evaluation"
        
        final_answer = answer_data
                
        system_prompt = """You are an expert mathematician tasked with evaluating the correctness of an answer to a mathematical question.
        
        Compare the generated answer to the ground truth answer and determine whether the generated answer is mathematically correct
        and equivalent to the ground truth.
        
        Please be very strict and rigorous in your evaluation, mark the answer as incorrect even if it is 80% or 90% correct.
        Ensure the generated answer can be directly rendered in standard LaTeX without requiring custom command definitions.
        Be precise and focus on mathematical correctness, not formatting or style differences.
        Your evaluation should be fair and consider that the same mathematical content can be expressed in different ways."""
        
        user_prompt = f"""QUESTION:
        {question}
        
        GROUND TRUTH ANSWER:
        {ground_truth}
        
        GENERATED ANSWER:
        {final_answer}
        
        Carefully evaluate whether the generated answer is mathematically correct 
        and equivalent to the ground truth. Your response should only contain a JSON object with the following fields:
        {{
          "is_correct": boolean,
          "explanation": "A concise explanation of why the answer is correct or incorrect, in a clean LaTeX format"
        }}
        where is_correct is true if the answer is mathematically correct and equivalent to the ground truth, and false if it isn't."""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            result = json.loads(response.choices[0].message.content)
            return result.get("is_correct", False), result.get("explanation", "No explanation provided")
        except Exception as e:
            if self.verbose:
                console.print(f"[bold red]Error evaluating answer: {e}[/bold red]")
            return False, f"Error evaluating: {str(e)}"
    
    def run_evaluation(self, dataset, model_name, use_context=True):
        """
        Run the evaluation on a dataset.
        
        Args:
            dataset: The dataset to evaluate on.
            model_name (str): The name of the model to evaluate.
            use_context (bool, optional): Whether to include context in the prompt. Defaults to True.
            
        Returns:
            dict: Evaluation results.
        """
        results = []
        correct_count = 0
        correct_ids = []  # Track IDs of correctly answered questions
        
        for i, example in enumerate(tqdm(dataset, desc=f"Evaluating {model_name}", disable=not self.verbose)):
            context = example["context"] if use_context else ""
            theorem_content = example["theorem"]
            question = example["question"]
            ground_truth = example["answer"]
            paper_link = example["paper_link"]
            
            # Query model for answer
            if self.verbose:
                console.print(f"\n[bold]Evaluating question {i+1}/{len(dataset)}[/bold]")
                console.print(f"[bold]Question:[/bold] {question}")
                console.print(f"[bold]Context:[/bold] {'[OMITTED]' if not use_context else context[:100] + '...' if len(context) > 100 else context}")
            
            # Call the model and measure response time
            start_time = time.time()
            answer_data = self.query_model(context, question, model_name, use_context=use_context)

            response_time = time.time() - start_time
            
            # Evaluate the generated answer
            is_correct, explanation = self.evaluate_answer(answer_data, ground_truth, question)
            if is_correct:
                correct_count += 1
                correct_ids.append(i)  # Add ID to correct_ids list
            
            # Store result
            result = {
                "question_id": i,
                "paper_link": paper_link,
                "theorem": theorem_content,
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": answer_data,
                "is_correct": is_correct,
                "explanation": explanation,
            }
            results.append(result)
            
            if self.verbose:
                # Use a panel to display the evaluation results
                result_color = "green" if is_correct else "red"
                console.print(
                    Panel(
                        f"[bold]Theorem:[/bold] {theorem_content[:150]}...\n\n"
                        f"[bold]Question:[/bold] {question}\n\n"
                        f"[bold]Ground truth:[/bold] {ground_truth}\n\n"
                        f"[bold]Final answer:[/bold] {answer_data[:150]}...\n\n"
                        f"[bold]{result_color}]Correct:[/bold] {is_correct}\n"
                        f"[bold]Explanation:[/bold] {explanation}\n",
                        title=f"Question {i+1} Result",
                        border_style=result_color
                    )
                )
        
        # Calculate overall metrics
        accuracy = correct_count / len(dataset) if len(dataset) > 0 else 0
        
        metrics = {
            "model": model_name,
            "dataset_size": len(dataset),
            "accuracy": accuracy,
            "correct_count": correct_count,
            "correct_ids": correct_ids,  # Add the list of correct IDs
            "context_used": use_context,
            "results": results
        }
        
        return metrics
    
    def save_results(self, metrics, output_path=None):
        """
        Save the evaluation results to a file.
        
        Args:
            metrics (dict): The evaluation metrics.
            output_path (str, optional): Path to save results to. Defaults to None.
            
        Returns:
            str: Path where results were saved.
        """
        if output_path is None:
            model_name = metrics["model"].replace("/", "-")
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = f"results_{model_name}_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        if self.verbose:
            console.print(f"[green]Results saved to {output_path}[/green]")
            
        return output_path


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Evaluate LLM performance on mathematical QA tasks")
    parser.add_argument("--dataset", type=str, default="ethz-spylab/arxiv_math_bench", 
                        help="Path to dataset or Hugging Face dataset ID. "
                             "Format: 'organization/dataset_name' for Hugging Face, or local path for disk storage.")
    # subset of the dataset
    parser.add_argument("--subset", type=str, choices=["math", "cs", "stackexchange"], default="stackexchange", 
                        help="Subset of the dataset to evaluate on. Available options: math, cs, stackexchange")
    parser.add_argument("--model", type=str, default="gpt-4o", 
                        help="Model to evaluate. Available options: " + 
                            ", ".join(list(OPENAI_MODELS.keys()) + list(ANTHROPIC_MODELS.keys()) + list(OPENROUTER_MODELS.keys())))
    parser.add_argument("--sample", type=int, help="Number of random samples to evaluate from the dataset")
    parser.add_argument("--output", type=str, help="Path to save results JSON file (default: auto-generated based on model and timestamp)")
    parser.add_argument("--verbose",type=bool, default=True, help="Print detailed information during processing")
    parser.add_argument("--no_context", action="store_true", help="Only provide the question to the LLM without context (tests raw knowledge)")
    parser.add_argument("--num_run", type=int, default=1, help="run the evaluation multiple times and average the results")

    
    args = parser.parse_args()
    
    
    console.print(
        Panel(
            "This tool evaluates LLM performance on mathematical question-answering tasks.\n"
            "It works by prompting the LLM with a mathematical problem and evaluating its answer against the ground truth.",
            title="Math QA Evaluator",
            border_style="blue"
        )
    )
    
    # Create an instance of MathQAEvaluator
    evaluator = MathQAEvaluator(
        verbose=args.verbose
    )
    
    # Load the dataset
    dataset = evaluator.load_dataset(args.dataset, sample_size=args.sample, subset=args.subset)
    if dataset is None:
        console.print(f"[bold red]Error: Could not load dataset from {args.dataset}[/bold red]")
        return
    
    console.print(f"[green]Successfully loaded dataset '{args.dataset}' with {len(dataset)} examples[/green]")
    
    # Single model evaluation mode
    console.print(f"\n[bold]Evaluating {args.model} on {len(dataset)} examples[/bold]")
    if args.no_context:
        console.print("[yellow]Mode: No context provided (testing raw LLM knowledge)[/yellow]")
    else:
        console.print("[green]Mode: Using provided context[/green]")
    
    # Run evaluation
    metrics = evaluator.run_evaluation(dataset, args.model, use_context=not args.no_context)
    
    # Print summary results
    console.print(
        Panel(
            f"[bold]Model:[/bold] {args.model}\n"
            f"[bold]Dataset size:[/bold] {metrics['dataset_size']}\n"
            f"[bold]Context used:[/bold] {metrics['context_used']}\n"
            f"[bold]Accuracy:[/bold] {metrics['accuracy']:.4f} ({metrics['correct_count']}/{metrics['dataset_size']})\n"
            f"[bold]Correct question IDs:[/bold] {metrics['correct_ids']}",
            title="Evaluation Results",
            border_style="green"
        )
    )
    
    # Save results
    if args.output:
        if not os.path.exists(args.output):
            os.makedirs(args.output, exist_ok=True)
        
        if args.no_context:
            output_path = f"{args.output}/{args.model}_wo_context_{args.num_run}run.jsonl"
        else:
            output_path = f"{args.output}/{args.model}_w_context_{args.num_run}run.jsonl"
        results_path = evaluator.save_results(metrics, output_path)
    else:
        # Default save behavior if no output path provided
        results_path = evaluator.save_results(metrics)
        
    console.print(f"\n[bold green]Detailed results saved to:[/bold green] {results_path}")
    

            
if __name__ == "__main__":
    main() 