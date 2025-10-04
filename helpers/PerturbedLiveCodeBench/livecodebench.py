import os
import json
import ast
import asyncio
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
from tqdm import tqdm
import openai
from datasets import load_dataset
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import subprocess
import tempfile
import re
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LiveCodeBenchEvaluator:
    def __init__(self, api_key: str):
        """
        Initialize the evaluator with OpenAI API key.
        
        Args:
            api_key: OpenAI API key
        """
        self.client = openai.OpenAI(api_key=api_key)
        self.dataset = None
        self.perturbed_problems = []
        self.results = defaultdict(lambda: defaultdict(list))
        self.detailed_results = []  # Store everything for debugging
        
    def load_livecodebench(self):
        """Load the LiveCodeBench dataset."""
        try:
            # Load LiveCodeBench dataset with correct path and version
            # release_v1 contains problems from May 2023 to Mar 2024 (400 problems)
            # which covers our target range
            self.dataset = load_dataset(
                "livecodebench/code_generation_lite", 
                version_tag="release_v1",
                trust_remote_code=True,
                split="test"
            )
            logger.info(f"Loaded LiveCodeBench with {len(self.dataset)} problems")
            
            # Filter for date range May 2023 - Feb 2024
            filtered_data = []
            for item in self.dataset:
                # LiveCodeBench has 'question_date' field
                date_field = item.get('question_date', item.get('date', item.get('created_at', '')))
                if date_field:
                    # Parse date and check range
                    try:
                        if isinstance(date_field, str):
                            problem_date = datetime.strptime(date_field[:10], '%Y-%m-%d')
                        else:
                            # Might be timestamp
                            problem_date = datetime.fromtimestamp(date_field)
                            
                        if datetime(2023, 5, 1) <= problem_date <= datetime(2024, 2, 29):
                            filtered_data.append(item)
                    except Exception as e:
                        logger.debug(f"Could not parse date {date_field}: {e}")
                        # Include if we can't parse date
                        filtered_data.append(item)
            
            if not filtered_data:
                # If no filtering worked, use all data
                logger.warning("Date filtering didn't work, using all problems")
                filtered_data = list(self.dataset)
            
            self.dataset = filtered_data
            logger.info(f"Using {len(self.dataset)} problems")
            
        except Exception as e:
            logger.error(f"Error loading LiveCodeBench: {e}")
            logger.info("Creating sample dataset for demonstration")
            self.dataset = self._create_sample_dataset()
    
    def _create_sample_dataset(self):
        """Create a sample dataset structure for demonstration if real dataset fails."""
        sample_problems = []
        months = pd.date_range('2023-05', '2024-02', freq='MS')
        
        for i, month in enumerate(months):
            for j in range(3):  # 3 problems per month for demo
                sample_problems.append({
                    'question_id': f'problem_{i}_{j}',
                    'question_date': month.strftime('%Y-%m-%d'),
                    'question_title': f'Two Sum Variation {i}_{j}',
                    'question_content': f'''Given an array of integers nums and an integer target, 
                    return indices of the two numbers such that they add up to target.
                    Example: nums = [{j}, {i}, {j+i}], target = {j+i}
                    Output: [0, 1]''',
                    'starter_code': 'def twoSum(nums, target):\n    pass',
                    'public_test_cases': json.dumps([
                        {'input': f'[{j}, {i}, {j+i}], {j+i}', 'output': '[0, 1]', 'testtype': 'function'},
                        {'input': f'[1, 2, 3], 5', 'output': '[1, 2]', 'testtype': 'function'}
                    ]),
                    'private_test_cases': json.dumps([
                        {'input': '[2, 7, 11, 15], 9', 'output': '[0, 1]', 'testtype': 'function'}
                    ]),
                    'difficulty': 'easy',
                    'metadata': json.dumps({'source': 'demo'})
                })
        
        return sample_problems
    
    def parse_test_cases(self, test_case_str: str) -> List[Dict]:
        """Parse test cases from JSON string format."""
        if not test_case_str:
            return []
        
        try:
            if isinstance(test_case_str, str):
                return json.loads(test_case_str)
            elif isinstance(test_case_str, list):
                return test_case_str
            else:
                return []
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Could not parse test cases: {e}")
            return []
    
    def extract_code_from_response(self, response: str) -> str:
        """Extract code from model response."""
        # Try to find code blocks
        if '```python' in response:
            matches = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
            if matches:
                return matches[0].strip()
        elif '```' in response:
            matches = re.findall(r'```\n(.*?)```', response, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # If no code blocks, return the whole response (it might be just code)
        return response.strip()
    
    def perturb_problem(self, problem_data: Dict) -> Dict:
        """
        Perturb a single problem using OpenAI's O1 model.
        
        Args:
            problem_data: Original problem data
            
        Returns:
            Problem data with perturbed version added
        """
        try:
            # Extract problem description
            problem_text = problem_data.get('question_content', 
                                           problem_data.get('problem_description', 
                                           problem_data.get('problem', '')))
            
            # Parse test cases
            public_tests = self.parse_test_cases(problem_data.get('public_test_cases', '[]'))
            private_tests = self.parse_test_cases(problem_data.get('private_test_cases', '[]'))
            
            # Create a simpler test case representation for the prompt
            test_examples = []
            for test in public_tests: 
                if isinstance(test, dict):
                    test_examples.append(f"Input: {test.get('input', '')}\nOutput: {test.get('output', '')}")
            
            prompt = f"""You are tasked with creating a variation of the following programming problem.

The variation should:
1. Keep the exact same algorithmic approach and complexity
2. Change variable names, function names, and context (e.g., if it uses 'abc', use something like 'XYZ')
3. Modify specific values in test cases consistently with the context change
4. Maintain the same difficulty level and logic

Original Problem:
{problem_text}

Original Test Examples:
{chr(10).join(test_examples)}

Provide the perturbed problem AND perturbed test cases in the following JSON format:
{{
    "problem_statement": "...",
    "test_cases": [
        {{"input": "...", "output": "...", "testtype": "stdin"}}
    ]
}}

Make sure to perturb ALL test values consistently. If the original uses 'abc', 'acb', 'bac', etc., 
and you change to 'XYZ', then use 'XYZ', 'XZY', 'YXZ', etc. correspondingly."""

            start_time = time.time()
            response = self.client.chat.completions.create(
                model="o1",  # Using O1-preview model
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=100000  # O1 uses max_completion_tokens
            )
            
            elapsed = time.time() - start_time
            logger.info(f"O1 perturbation took {elapsed:.1f} seconds")
            
            response_text = response.choices[0].message.content.strip()
            
            # Try to parse the JSON response
            try:
                # Extract JSON from response if wrapped in code blocks
                if '```json' in response_text:
                    json_match = re.search(r'```json\n(.*?)```', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(1)
                elif '```' in response_text:
                    json_match = re.search(r'```\n(.*?)```', response_text, re.DOTALL)
                    if json_match:
                        response_text = json_match.group(1)
                
                perturbed_data = json.loads(response_text)
                
                # Ensure we have the required fields
                perturbed_problem = perturbed_data.get('problem_statement', response_text)
                perturbed_test_cases = perturbed_data.get('test_cases', [])
                
                # Make sure we have enough perturbed test cases
                # If not, create modified versions based on the pattern
                total_tests_needed = len(public_tests) + len(private_tests)
               # if len(perturbed_test_cases) < total_tests_needed:
                #    logger.warning(f"Only got {len(perturbed_test_cases)} perturbed tests, needed {total_tests_needed}")
                
            except (json.JSONDecodeError, Exception) as e:
                logger.warning(f"Could not parse perturbed response as JSON: {e}")
                # Fall back to using the response as the problem statement
                perturbed_problem = response_text
                perturbed_test_cases = []
                
                # Try to create perturbed test cases by simple substitution
                # This is a fallback - ideally O1 should provide them
                if 'abc' in problem_text.lower() and any('XYZ' in s for s in [response_text]):
                    # Simple character substitution for test cases
                    mapping = {'a': 'X', 'b': 'Y', 'c': 'Z'}
                    for test in public_tests + private_tests:
                        if isinstance(test, dict):
                            new_input = test.get('input', '')
                            new_output = test.get('output', '')
                            for old, new in mapping.items():
                                new_input = new_input.replace(old, new)
                                new_output = new_output.replace(old, new)
                            perturbed_test_cases.append({
                                'input': new_input,
                                'output': new_output,
                                'testtype': test.get('testtype', 'stdin')
                            })
            
            # Split perturbed test cases back into public and private
            num_public = len(public_tests)
            perturbed_public = perturbed_test_cases[:num_public] if perturbed_test_cases else public_tests
            perturbed_private = perturbed_test_cases[num_public:] if len(perturbed_test_cases) > num_public else private_tests
            
            return {
                **problem_data,
                'perturbed_problem': perturbed_problem,
                'perturbed_public_test_cases': perturbed_public,
                'perturbed_private_test_cases': perturbed_private,
                'perturbation_time': elapsed
            }
            
        except Exception as e:
            logger.error(f"Error perturbing problem: {e}")
            # Return original if perturbation fails
            return {
                **problem_data,
                'perturbed_problem': problem_text,
                'perturbed_public_test_cases': public_tests,
                'perturbed_private_test_cases': private_tests,
                'perturbation_error': str(e)
            }
    
    def perturb_all_problems(self, sample_size: Optional[int] = None):
        """
        Perturb all problems in the dataset using O1.
        
        Args:
            sample_size: Optional limit on number of problems to perturb
        """
        problems = self.dataset
        if True:
            problems = problems
        
        logger.info(f"Perturbing {len(problems)} problems with O1...")
        
        for problem in tqdm(problems, desc="Perturbing with O1"):
            perturbed = self.perturb_problem(problem)
            self.perturbed_problems.append(perturbed)
            
            # O1 has strict rate limits, add delay
            time.sleep(2)  # Adjust based on your rate limits
    
    def solve_problem(self, problem_text: str, model: str, original_problem_data: Dict) -> Tuple[str, str, float]:
        """
        Solve a problem using specified GPT model.
        
        Args:
            problem_text: Problem statement (perturbed)
            model: Model name ('gpt-4' or 'gpt-4o')
            original_problem_data: Original problem data for function signature
            
        Returns:
            Tuple of (generated solution code, raw response, time taken)
        """
        try:
            # Check if this is a stdin/stdout problem or a function problem
            public_tests = self.parse_test_cases(original_problem_data.get('public_test_cases', '[]'))
            perturbed_tests = original_problem_data.get('perturbed_public_test_cases', public_tests)
            
            is_stdin_problem = False
            if perturbed_tests and isinstance(perturbed_tests[0], dict):
                is_stdin_problem = perturbed_tests[0].get('testtype') == 'stdin'
            
            # Build the prompt
            prompt_parts = [
                "Solve the following programming problem.",
                "Provide a complete, working Python solution.",
                "",
                "Problem:",
                problem_text,
                ""
            ]
            
            # Add guidance based on problem type
            if is_stdin_problem:
                prompt_parts.append("This is a standard competitive programming problem that reads from stdin and writes to stdout.")
                prompt_parts.append("Your solution should use input() to read data and print() to output results.")
            else:
                # Extract function signature if available
                starter_code = original_problem_data.get('starter_code', '')
                entry_point = original_problem_data.get('entry_point', '')
                
                if starter_code:
                    prompt_parts.append(f"Use this function signature:")
                    prompt_parts.append(starter_code)
                elif entry_point:
                    prompt_parts.append(f"The main function should be named: {entry_point}")
            
            # Add example from perturbed test cases if available
            if perturbed_tests:
                prompt_parts.append("\nExample:")
                test_example = perturbed_tests[0]
                if isinstance(test_example, dict):
                    prompt_parts.append(f"Input: {test_example.get('input', '')}")
                    prompt_parts.append(f"Output: {test_example.get('output', '')}")
            
            prompt_parts.append("")
            prompt_parts.append("Provide the complete Python code solution. Do not include any explanations, just the code:")
            
            # Join all parts with newline
            prompt = '\n'.join(prompt_parts)

            # Map model names to actual OpenAI model identifiers
            model_mapping = {
                'gpt-4': 'gpt-4-turbo',  # Latest GPT-4 Turbo
                'gpt-4o': 'gpt-4o'  # GPT-4o
            }
            
            actual_model = model_mapping.get(model, model)
            
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=actual_model,
                messages=[
                    {"role": "system", "content": "You are an expert competitive programmer. Provide complete, efficient, and correct Python code solutions. Output only code, no explanations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Use 0 for deterministic output
                max_tokens=100000  # Much higher limit for complex solutions
            )
            
            elapsed = time.time() - start_time
            logger.debug(f"{model} took {elapsed:.1f} seconds")
            
            raw_response = response.choices[0].message.content
            solution = self.extract_code_from_response(raw_response)
            
            return solution, raw_response, elapsed
            
        except Exception as e:
            logger.error(f"Error solving problem with {model}: {e}")
            return "", f"Error: {str(e)}", 0.0
    
    def evaluate_solution(self, solution: str, problem_data: Dict) -> Tuple[bool, str, List[Dict]]:
        """
        Evaluate a solution against test cases from LiveCodeBench.
        
        Args:
            solution: Generated solution code
            problem_data: Problem data including test cases
            
        Returns:
            Tuple of (success, error_message, test_results)
        """
        if not solution:
            return False, "No solution provided", []
        
        test_results = []
        
        try:
            # Get the perturbed test cases
            public_tests = problem_data.get('perturbed_public_test_cases', [])
            private_tests = problem_data.get('perturbed_private_test_cases', [])
            
            # Fallback to original test cases if no perturbed ones
            if not public_tests and not private_tests:
                public_tests = self.parse_test_cases(problem_data.get('public_test_cases', '[]'))
                private_tests = self.parse_test_cases(problem_data.get('private_test_cases', '[]'))
            
            all_tests = public_tests + private_tests
            
            if not all_tests:
                logger.warning("No test cases found in problem data")
                return False, "No test cases found", []
            
            # Determine test type
            is_stdin_problem = False
            if all_tests and isinstance(all_tests[0], dict):
                is_stdin_problem = all_tests[0].get('testtype') == 'stdin'
            
            # Create temporary file to run the solution
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                temp_file = f.name
            
            passed_count = 0
            
            for i, test in enumerate(all_tests):
                if not isinstance(test, dict):
                    test_results.append({'test': i, 'result': 'SKIP', 'error': 'Invalid test format'})
                    continue
                
                test_input = test.get('input', '')
                expected_output = test.get('output', '')
                
                try:
                    if is_stdin_problem:
                        # Write the solution and run with stdin
                        with open(temp_file, 'w') as f:
                            f.write(solution)
                        
                        # Run with the input
                        result = subprocess.run(
                            ['python', temp_file],
                            input=test_input,
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        
                        actual_output = result.stdout.strip()
                        
                    else:
                        # Function-based problem
                        # Try to extract function name from solution
                        func_match = re.search(r'def\s+(\w+)\s*\(', solution)
                        if not func_match:
                            test_results.append({'test': i, 'result': 'FAIL', 'error': 'No function found'})
                            continue
                        
                        func_name = func_match.group(1)
                        
                        # Write solution with test code
                        with open(temp_file, 'w') as f:
                            f.write(solution)
                            f.write(f'\n\n# Test execution\n')
                            f.write(f'import json\n')
                            f.write(f'try:\n')
                            f.write(f'    result = {func_name}({test_input})\n')
                            f.write(f'    print(json.dumps(result))\n')
                            f.write(f'except Exception as e:\n')
                            f.write(f'    print(f"ERROR: {{e}}")\n')
                        
                        result = subprocess.run(
                            ['python', temp_file],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        
                        actual_output = result.stdout.strip()
                        
                        # Try to parse JSON output for function results
                        if actual_output and not actual_output.startswith("ERROR"):
                            try:
                                actual_output = json.loads(actual_output)
                                actual_output = str(actual_output)
                            except:
                                pass
                    
                    # Compare outputs
                    if actual_output == expected_output:
                        test_results.append({'test': i, 'result': 'PASS'})
                        passed_count += 1
                    else:
                        test_results.append({
                            'test': i, 
                            'result': 'FAIL',
                            'expected': expected_output[:100],
                            'actual': actual_output[:100],
                            'error': result.stderr[:200] if result.stderr else None
                        })
                        
                except subprocess.TimeoutExpired:
                    test_results.append({'test': i, 'result': 'TIMEOUT'})
                except Exception as e:
                    test_results.append({'test': i, 'result': 'ERROR', 'error': str(e)[:200]})
            
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
            
            success = passed_count == len(all_tests)
            return success, f"{passed_count}/{len(all_tests)} tests passed", test_results
            
        except Exception as e:
            logger.error(f"Evaluation error: {e}\n{traceback.format_exc()}")
            return False, f"Evaluation error: {str(e)}", []
    
    def run_evaluation(self, models: List[str] = ['gpt-4', 'gpt-4o'], 
                       sample_size: Optional[int] = None):
        """
        Run the complete evaluation pipeline.
        
        Args:
            models: List of models to evaluate
            sample_size: Optional limit on number of problems to evaluate
        """
        # Load dataset if not already loaded
        if self.dataset is None:
            self.load_livecodebench()
        
        # Perturb problems if not already done
        if not self.perturbed_problems:
            self.perturb_all_problems()
        
        # Evaluate each model on perturbed problems
        for model in models:
            logger.info(f"Evaluating {model}...")
            
            for problem in tqdm(self.perturbed_problems, desc=f"Evaluating {model}"):
                # Get problem date and extract month-year
                date_str = problem.get('question_date', problem.get('date', problem.get('created_at', '')))
                
                if date_str:
                    try:
                        if isinstance(date_str, str):
                            problem_date = datetime.strptime(date_str[:10], '%Y-%m-%d')
                        else:
                            problem_date = datetime.fromtimestamp(date_str)
                        month_key = problem_date.strftime('%Y-%m')
                    except:
                        month_key = '2023-12'
                else:
                    month_key = '2023-12'
                
                # Get problem ID
                problem_id = problem.get('question_id', problem.get('id', problem.get('problem_id', 'unknown')))
                
                # Solve the perturbed problem
                solution, raw_response, solve_time = self.solve_problem(
                    problem['perturbed_problem'], 
                    model,
                    problem
                )
                
                # Evaluate the solution using perturbed test cases
                is_correct, error_msg, test_results = self.evaluate_solution(solution, problem)
                
                # Store detailed results
                detailed_result = {
                    'problem_id': problem_id,
                    'model': model,
                    'month': month_key,
                    'correct': is_correct,
                    'solution': solution,
                    'raw_response': raw_response,
                    'error_message': error_msg,
                    'test_results': test_results,
                    'solve_time': solve_time,
                    'original_problem_data': {
                        'question_id': problem.get('question_id'),
                        'question_date': problem.get('question_date'),
                        'question_title': problem.get('question_title'),
                        'question_content': problem.get('question_content', problem.get('problem', '')),
                        'starter_code': problem.get('starter_code'),
                        'entry_point': problem.get('entry_point'),
                        'public_test_cases': problem.get('public_test_cases', '[]'),
                        'private_test_cases': problem.get('private_test_cases', '[]'),
                        'difficulty': problem.get('difficulty'),
                        'metadata': problem.get('metadata', '{}')
                    },
                    'perturbed_problem': problem['perturbed_problem'],
                    'perturbed_public_test_cases': problem.get('perturbed_public_test_cases', []),
                    'perturbed_private_test_cases': problem.get('perturbed_private_test_cases', []),
                    'perturbation_time': problem.get('perturbation_time'),
                    'perturbation_error': problem.get('perturbation_error')
                }
                
                self.detailed_results.append(detailed_result)
                
                # Store summary results
                self.results[model][month_key].append({
                    'problem_id': problem_id,
                    'correct': is_correct
                })
                
                # Rate limiting between API calls
                time.sleep(1)
    
    def calculate_accuracy(self) -> pd.DataFrame:
        """
        Calculate accuracy for each model by month.
        
        Returns:
            DataFrame with accuracy metrics
        """
        accuracy_data = []
        
        for model in self.results:
            for month in sorted(self.results[model].keys()):
                month_results = self.results[model][month]
                if month_results:
                    accuracy = sum(r['correct'] for r in month_results) / len(month_results)
                    accuracy_data.append({
                        'model': model,
                        'month': month,
                        'accuracy': accuracy * 100,  # Convert to percentage
                        'total_problems': len(month_results),
                        'correct_problems': sum(r['correct'] for r in month_results)
                    })
        
        df = pd.DataFrame(accuracy_data)
        if not df.empty:
            df['month'] = pd.to_datetime(df['month'] + '-01')
            df = df.sort_values(['model', 'month'])
        
        return df
    
    def save_results(self, filepath: str = 'evaluation_results.json'):
        """
        Save complete evaluation results including all intermediate outputs.
        
        Args:
            filepath: Path to save results
        """
        # Calculate summary
        summary_df = self.calculate_accuracy()
        
        # Create problems collection with original and perturbed versions
        problems_collection = []
        unique_problems = {}
        
        # Collect unique problems from detailed results
        for result in self.detailed_results:
            problem_id = result['problem_id']
            if problem_id not in unique_problems:
                unique_problems[problem_id] = {
                    'problem_id': problem_id,
                    'original_problem': result['original_problem_data'],
                    'perturbed_problem': result['perturbed_problem'],
                    'perturbed_public_test_cases': result.get('perturbed_public_test_cases', []),
                    'perturbed_private_test_cases': result.get('perturbed_private_test_cases', []),
                    'perturbation_time': result.get('perturbation_time'),
                    'perturbation_error': result.get('perturbation_error'),
                    'solutions': {}
                }
            
            # Add solution for this model
            model = result['model']
            unique_problems[problem_id]['solutions'][model] = {
                'solution_code': result['solution'],
                'raw_response': result['raw_response'],
                'correct': result['correct'],
                'error_message': result['error_message'],
                'test_results': result['test_results'],
                'solve_time': result['solve_time']
            }
        
        problems_collection = list(unique_problems.values())
        
        results_to_save = {
            'summary': summary_df.to_dict('records') if not summary_df.empty else [],
            'problems': problems_collection,
            'detailed_results': self.detailed_results,
            'metadata': {
                'evaluation_date': datetime.now().isoformat(),
                'total_problems': len(self.perturbed_problems),
                'models_evaluated': list(self.results.keys())
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        logger.info(f"Complete results saved to {filepath}")
        
        # Also save a separate file with just problems and solutions for easy access
        problems_and_solutions_file = 'problems_and_solutions.json'
        with open(problems_and_solutions_file, 'w') as f:
            json.dump({'problems': problems_collection}, f, indent=2, default=str)
        logger.info(f"Problems and solutions saved to {problems_and_solutions_file}")
        
        # Also save a separate debugging file with just the failed cases
        failed_cases = [r for r in self.detailed_results if not r['correct']]
        if failed_cases:
            with open('failed_cases.json', 'w') as f:
                json.dump(failed_cases, f, indent=2, default=str)
            logger.info(f"Failed cases saved to failed_cases.json ({len(failed_cases)} failures)")
    
    def plot_results(self):
        """Generate and display accuracy plots."""
        import matplotlib.pyplot as plt
        
        df = self.calculate_accuracy()
        
        if df.empty:
            logger.warning("No results to plot")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Accuracy over time
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            ax1.plot(model_data['month'], model_data['accuracy'], 
                    marker='o', label=model, linewidth=2, markersize=8)
        
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Model Accuracy on Perturbed LiveCodeBench Problems (May 2023 - Feb 2024)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # Format x-axis
        import matplotlib.dates as mdates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # Plot 2: Bar chart comparing total accuracy
        model_names = df['model'].unique()
        avg_accuracies = [df[df['model'] == m]['accuracy'].mean() for m in model_names]
        
        bars = ax2.bar(model_names, avg_accuracies, color=['blue', 'green'])
        ax2.set_ylabel('Average Accuracy (%)')
        ax2.set_title('Overall Average Accuracy Comparison')
        ax2.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, val in zip(bars, avg_accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom')
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('accuracy_plot.png', dpi=150)
        plt.show()
        
        logger.info("Plot saved as accuracy_plot.png")


def main():
    # Configuration
    API_KEY = os.environ.get('OPENAI_API_KEY')
    
    if API_KEY == 'your-api-key-here':
        logger.error("Please set your OpenAI API key in the OPENAI_API_KEY environment variable")
        return
    
    # Initialize evaluator
    evaluator = LiveCodeBenchEvaluator(API_KEY)
    
    # Run evaluation
    # Start with small sample for testing, then increase
    evaluator.run_evaluation(
        models=['gpt-4', 'gpt-4o'],
        sample_size=5  # Start small for testing, then remove this limit
    )
    
    # Calculate and display accuracy
    accuracy_df = evaluator.calculate_accuracy()
    print("\n" + "="*60)
    print("ACCURACY RESULTS BY MONTH")
    print("="*60)
    if not accuracy_df.empty:
        for model in accuracy_df['model'].unique():
            print(f"\n{model}:")
            model_data = accuracy_df[accuracy_df['model'] == model]
            for _, row in model_data.iterrows():
                print(f"  {row['month'].strftime('%b %Y')}: {row['accuracy']:.1f}% "
                      f"({row['correct_problems']}/{row['total_problems']} correct)")
    
    # Save complete results with debugging info
    evaluator.save_results()
    
    # Plot results
    try:
        evaluator.plot_results()
    except ImportError:
        logger.warning("Matplotlib not available for plotting")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    if not accuracy_df.empty:
        for model in accuracy_df['model'].unique():
            model_data = accuracy_df[accuracy_df['model'] == model]
            print(f"\n{model}:")
            print(f"  Average accuracy: {model_data['accuracy'].mean():.1f}%")
            print(f"  Min accuracy: {model_data['accuracy'].min():.1f}%")
            print(f"  Max accuracy: {model_data['accuracy'].max():.1f}%")
            print(f"  Total problems evaluated: {model_data['total_problems'].sum()}")
            print(f"  Total correct: {model_data['correct_problems'].sum()}")


if __name__ == "__main__":
    main()
