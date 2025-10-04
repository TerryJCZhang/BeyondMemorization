import json
import numpy as np
from datetime import datetime
from collections import defaultdict

def analyze_json_results(filepath='evaluation_results.json'):
    """
    Analyze the JSON file to extract per-month accuracy for both GPT-4 and GPT-4o.
    
    Args:
        filepath: Path to the JSON results file
        
    Returns:
        dict: Contains arrays of accuracy values for each model
    """
    # Load the JSON file
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Initialize result structure
    results = {
        'gpt-4': {},
        'gpt-4o': {}
    }
    
    # Extract accuracy from summary
    if 'summary' in data:
        for record in data['summary']:
            model = record['model']
            month = record['month']
            accuracy = record['accuracy']
            
            # Convert month string to datetime for proper sorting
            if isinstance(month, str):
                # Handle different date formats
                try:
                    # Try parsing as full datetime
                    month_dt = datetime.strptime(month[:7], '%Y-%m')
                except:
                    try:
                        # Try parsing if already in YYYY-MM format
                        month_dt = datetime.strptime(month, '%Y-%m')
                    except:
                        # Skip if can't parse
                        continue
                
                month_key = month_dt.strftime('%Y-%m')
                
                if model in results:
                    results[model][month_key] = accuracy
    
    # Alternative: Calculate from detailed_results if summary is empty
    if not any(results[m] for m in results):
        print("Summary empty, calculating from detailed results...")
        
        monthly_stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
        
        for result in data.get('detailed_results', []):
            model = result['model']
            month = result['month']
            is_correct = result['correct']
            
            if model in results:
                monthly_stats[model][month]['total'] += 1
                if is_correct:
                    monthly_stats[model][month]['correct'] += 1
        
        # Calculate accuracies
        for model in monthly_stats:
            for month in monthly_stats[model]:
                stats = monthly_stats[model][month]
                if stats['total'] > 0:
                    accuracy = (stats['correct'] / stats['total']) * 100
                    results[model][month] = accuracy
    
    # Sort by month and create arrays
    months_order = ['2023-05', '2023-06', '2023-07', '2023-08', '2023-09', 
                    '2023-10', '2023-11', '2023-12', '2024-01', '2024-02']
    
    gpt4_accuracies = []
    gpt4o_accuracies = []
    
    for month in months_order:
        # Get accuracy for each model, default to 0 if not found
        gpt4_accuracies.append(results['gpt-4'].get(month, 0))
        gpt4o_accuracies.append(results['gpt-4o'].get(month, 0))
    
    # Convert to numpy arrays
    gpt4_array = np.array(gpt4_accuracies)
    gpt4o_array = np.array(gpt4o_accuracies)
    
    # Print results for verification
    print("="*60)
    print("EXTRACTED ACCURACY DATA")
    print("="*60)
    print("\nGPT-4 Accuracies by month:")
    for i, month in enumerate(months_order):
        if gpt4_accuracies[i] > 0:
            print(f"  {month}: {gpt4_accuracies[i]:.2f}%")
    
    print("\nGPT-4o Accuracies by month:")
    for i, month in enumerate(months_order):
        if gpt4o_accuracies[i] > 0:
            print(f"  {month}: {gpt4o_accuracies[i]:.2f}%")
    
    print("\n" + "="*60)
    print("Arrays ready for plotting:")
    print(f"gpt4_percentages = {gpt4_array}")
    print(f"gpt4o_percentages = {gpt4o_array}")
    print("="*60)
    
    return {
        'gpt4': gpt4_array,
        'gpt4o': gpt4o_array,
        'months': months_order
    }


if __name__ == "__main__":
    # Analyze the JSON file and extract accuracy arrays
    print("Analyzing evaluation results from 'evaluation_results.json'...")
    
    accuracy_data = analyze_json_results('evaluation_results.json')
    
    # Store the arrays in variables that can be used by the plotting code
    gpt4_percentages = accuracy_data['gpt4']
    gpt4o_percentages = accuracy_data['gpt4o']
    
    print("\n✓ Accuracy arrays extracted successfully!")
    print("Variables 'gpt4_percentages' and 'gpt4o_percentages' are now available for plotting.")
