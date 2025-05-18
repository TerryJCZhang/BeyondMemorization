from calendar import month_name
from itertools import tee
import json
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib_venn import venn2, venn3
import argparse
import warnings
warnings.filterwarnings("ignore")
import matplotlib.dates as mdates
from adjustText import adjust_text
from matplotlib.ticker import PercentFormatter
import arxiv
import requests
import time
import matplotlib

# Set seaborn style for nicer plots
sns.set(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'],
    'axes.labelsize': 18,
    'axes.titlesize': 18,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18
})

SUBSET = {
    "math_arxiv": "/data/projects/math_bench/final_results/results_math",
    "cs_arxiv": "/data/projects/math_bench/final_results/results_cs",
    "math_stackexchange": "/data/projects/math_bench/final_results/results_stackexchange",
    "math_arxiv_ft": "/data/projects/math_bench/final_results/results_math_ft",
    "math_arxiv_hardest": "/data/projects/math_bench/difficulty/math_three_groups",
}

MODEL_RELEASE_DATE = {
    "o3": "2025-04-16",
    "o4-mini": "2025-04-16",
    "gemini-2.5-pro": "2025-03-25",
    "grok-3": "2025-02-17",
    "deepseek-r1": "2025-01-10",
    "claude-3.7-sonnet": "2025-02-19",
    "qwen3-235b": "2025-04-29",
    "gpt-4o-mini": "2024-07-18",
    "llama-3.1-405b": "2024-07-23",
    "claude-3.5-sonnet": "2024-06-20",
}

UPPERCASE_MODELS = {
    "gpt-4o-mini": "GPT-4o-mini",
    "llama-3.1-405b": "Llama-3.1-405B",
    "claude-3.5-sonnet": "Claude-3.5-Sonnet",
    "claude-3.7-sonnet": "Claude-3.7-Sonnet",
    "qwen3-235b": "Qwen3-235B",
    "deepseek-r1": "DeepSeek-R1",
    "gemini-2.5-pro": "Gemini-2.5-Pro",
    "grok-3": "Grok-3",
    "o3": "o3",
    "o4-mini": "o4-mini",
}

# Mapping of arXiv category codes to full names
ARXIV_CATEGORY_NAMES = {
    # Mathematics categories
    "math.AG": "Algebraic Geometry",
    "math.AT": "Algebraic Topology",
    "math.AP": "Analysis of PDEs",
    "math.CT": "Category Theory",
    "math.CA": "Classical Analysis and ODEs",
    "math.CO": "Combinatorics",
    "math.AC": "Commutative Algebra",
    "math.CV": "Complex Variables",
    "math.DG": "Differential Geometry",
    "math.DS": "Dynamical Systems",
    "math.FA": "Functional Analysis",
    "math.GM": "General Mathematics",
    "math.GN": "General Topology",
    "math.GT": "Geometric Topology",
    "math.GR": "Group Theory",
    "math.HO": "History and Overview",
    "math.IT": "Information Theory",
    "math.KT": "K-Theory and Homology",
    "math.LO": "Logic",
    "math.MP": "Mathematical Physics",
    "math.MG": "Metric Geometry",
    "math.NT": "Number Theory",
    "math.NA": "Numerical Analysis",
    "math.OA": "Operator Algebras",
    "math.OC": "Optimization and Control",
    "math.PR": "Probability",
    "math.QA": "Quantum Algebra",
    "math.RT": "Representation Theory",
    "math.RA": "Rings and Algebras",
    "math.SP": "Spectral Theory",
    "math.ST": "Statistics Theory",
    "math.SG": "Symplectic Geometry",
    
    # Physics categories
    "physics.acc-ph": "Accelerator Physics",
    "physics.app-ph": "Applied Physics",
    "physics.ao-ph": "Atmospheric and Oceanic Physics",
    "physics.atom-ph": "Atomic Physics",
    "physics.atm-clus": "Atomic and Molecular Clusters",
    "physics.bio-ph": "Biological Physics",
    "physics.chem-ph": "Chemical Physics",
    "physics.class-ph": "Classical Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.data-an": "Data Analysis, Statistics and Probability",
    "physics.flu-dyn": "Fluid Dynamics",
    "physics.gen-ph": "General Physics",
    "physics.geo-ph": "Geophysics",
    "physics.hist-ph": "History and Philosophy of Physics",
    "physics.ins-det": "Instrumentation and Detectors",
    "physics.med-ph": "Medical Physics",
    "physics.optics": "Optics",
    "physics.ed-ph": "Physics Education",
    "physics.soc-ph": "Physics and Society",
    "physics.plasm-ph": "Plasma Physics",
    "physics.pop-ph": "Popular Physics",
    "physics.space-ph": "Space Physics",
    
    # Computer Science categories
    "cs.AI": "Artificial Intelligence",
    "cs.AR": "Hardware Architecture",
    "cs.CC": "Computational Complexity",
    "cs.CE": "Computational Engineering",
    "cs.CG": "Computational Geometry",
    "cs.CL": "Computation and Language",
    "cs.CR": "Cryptography and Security",
    "cs.CV": "Computer Vision and Pattern Recognition",
    "cs.CY": "Computers and Society",
    "cs.DB": "Databases",
    "cs.DC": "Distributed, Parallel, and Cluster Computing",
    "cs.DL": "Digital Libraries",
    "cs.DM": "Discrete Mathematics",
    "cs.DS": "Data Structures and Algorithms",
    "cs.ET": "Emerging Technologies",
    "cs.FL": "Formal Languages and Automata Theory",
    "cs.GL": "General Literature",
    "cs.GR": "Graphics",
    "cs.GT": "Computer Science and Game Theory",
    "cs.HC": "Human-Computer Interaction",
    "cs.IR": "Information Retrieval",
    "cs.IT": "Information Theory",
    "cs.LG": "Machine Learning",
    "cs.LO": "Logic in Computer Science",
    "cs.MA": "Multiagent Systems",
    "cs.MM": "Multimedia",
    "cs.MS": "Mathematical Software",
    "cs.NA": "Numerical Analysis",
    "cs.NE": "Neural and Evolutionary Computing",
    "cs.NI": "Networking and Internet Architecture",
    "cs.OH": "Other Computer Science",
    "cs.OS": "Operating Systems",
    "cs.PF": "Performance",
    "cs.PL": "Programming Languages",
    "cs.RO": "Robotics",
    "cs.SC": "Symbolic Computation",
    "cs.SD": "Sound",
    "cs.SE": "Software Engineering",
    "cs.SI": "Social and Information Networks",
    "cs.SY": "Systems and Control",
    
    # Main categories
    "math": "Mathematics",
    "cs": "Computer Science",
    "physics": "Physics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "stat": "Statistics",
    "econ": "Economics",
    "eess": "Electrical Engineering and Systems Science"
}

"""
models' release dates and knowledge cutoff dates:

o3: o3-2025-04-16
o4-mini: 2025-04-16
gemini-2.5-pro: 2025-03-25
grok-3: 2025-02-17
deepseek-r1: 2025-01-10
claude-3.7-sonnet: 2025-02-19
qwen3-235b: 2025-04-29


some old models:
gpt-4o-mini: 2024-07-18, cutoff date: 2023-10
llama-3.1-405b: 2024-07-23, 
claude-3.5-sonnet: 2024-06-20, cutoff date: 2024-04



"""


def analyze_papers(file_name=None, paper_links=None, bad_theorems=None, stats_math_arxiv=None, totals_math_arxiv=None, stats_cs_arxiv=None, totals_cs_arxiv=None):
    
    def get_paper_links(file_name):
        paper_links = []
        with open(file_name, 'r') as f:
            lines = [line.strip() for line in f]
        for line in lines:
            if 'url' in line:
                link = line.split('url')[-1][1:-1]
                paper_links.append(link)
        return paper_links

    def get_id(paper_link):
        parts = paper_link.split('/')
        id = parts[-1]
        return id

    def get_id_stack(paper_link):
        parts = paper_link.split('/')
        id = parts[4]
        return id

    def compute_stats(paper_links, bad):
        theorems = dict()
        bad_theorems = dict()
        stats = dict()
        for i in range(len(paper_links)):
            print(i)
            link = paper_links[i]
            arxiv_id = get_id(link)
            search = arxiv.Search(id_list=[arxiv_id])
            client = arxiv.Client(
                    page_size=100,  # Number of results per query
                    delay_seconds=3,  # Be nice to the API
                    num_retries=5    # Retry on failure
            )
            result = next(client.results(search), None)
            subcategories = []
            if result:
                subcategories = result.primary_category.split()
                for subcat in subcategories:
                    if subcat not in theorems:
                        theorems[subcat] = 0
                        bad_theorems[subcat] = 0
                    theorems[subcat] += 1
                    if i in bad:
                        bad_theorems[subcat] += 1
        for subcategory in theorems:
            stats[subcategory] = bad_theorems[subcategory] / theorems[subcategory]
        return stats, theorems

    def compute_stats_stack(paper_links, bad):
        theorems = dict()
        bad_theorems = dict()
        stats = dict()
        for i in range(1000):
            print(i)
            tags = []
            link = paper_links[i]
            id = get_id_stack(link)
            api_url = f"https://api.stackexchange.com/2.3/questions/{id}"
            params = {
                    'order': 'desc',
                    'sort': 'activity',
                    'site': 'math',
                }
            try:
                response = requests.get(api_url, params=params)
                data = response.json()
                if 'items' in data and data['items']:
                    tags =  data['items'][0]['tags']
                    for subcat in tags:
                        if subcat not in theorems:
                            theorems[subcat] = 0
                            bad_theorems[subcat] = 0
                        theorems[subcat] += 1
                        if i in bad:
                            bad_theorems[subcat] += 1
                time.sleep(1)
            except:
                print('Bad ' + str(i))
        for subcategory in theorems:
            stats[subcategory] = bad_theorems[subcategory] / theorems[subcategory]
        return stats

    def plot_graph(data, total, type='math'):
        keys = []
        values = []
        for category in data:
            if data[category] != 0:
                if type == 'cs':
                    if total[category] >= 5:
                        keys.append(category)
                        values.append(data[category] * 100)
                elif type == 'math':
                    if total[category] > 19 and category != ' Information Theory (cs.IT)':
                        keys.append(category)
                        values.append(data[category] * 100)

        plt.rcParams['font.family'] = 'serif'        
        plt.rcParams['font.serif'] = ['Times New Roman']  
        plt.rcParams['font.size'] = 12

        colors = sns.color_palette("pastel")

        plt.figure(figsize=(8, 5))
        bars = plt.barh(keys, values, color=colors)

        for bar, acc in zip(bars, values):
            width = bar.get_width()
            plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2, f"{acc:.1f}%",
                     ha='left', va='center', fontsize=10, weight='bold')

        plt.xlabel("Removed low-quality samples (%)")
        plt.ylabel("Category")

        ax = plt.gca()  
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)  
        ax.spines['bottom'].set_visible(True)

        if type == 'math':
            plt.xlim(0, 25)  
        elif type == 'cs':
            plt.xlim(0, 20)

        plt.gca().invert_yaxis()  

        plt.tight_layout()
        plt.savefig('bar_with_labels_' + type + '.png', dpi=300, bbox_inches='tight')
        plt.show()

    if file_name and paper_links is None:
        paper_links = get_paper_links(file_name)
    if bad_theorems is None:
        bad_theorems = []

    if paper_links:
        stats, totals = compute_stats(paper_links, bad_theorems)
        plot_graph(stats, totals, type='math')
    
    if stats_math_arxiv and totals_math_arxiv:
        plot_graph(stats_math_arxiv, totals_math_arxiv, type='math')
    
    if stats_cs_arxiv and totals_cs_arxiv:
        plot_graph(stats_cs_arxiv, totals_cs_arxiv, type='cs')

# # Example usage
# if __name__ == '__main__':
#     analyze_papers(file_name='arxiv_cs.txt', bad_theorems=[0, 4, 30, 34, 63, 70, 73, 80, 110])



def plot_per_category_math_arxiv():
    """
    Creates a vertical bar chart comparing two models across math arXiv categories.
    Applies consistent styling with other plotting functions.
    """
    # Define file paths for both models
    file1 = "/data/projects/math_bench/analyze/category_statistics_o3-results_annotated_math_level_arxiv_math_math_categories.json"
    file2 = "/data/projects/math_bench/analyze/category_statistics_gemini-2_annotated_math_level_arxiv_math_math_categories.json"

    # Extract model names from filenames for the legend
    model1_name = UPPERCASE_MODELS.get("o3", "o3") 
    model2_name = UPPERCASE_MODELS.get("gemini-2.5-pro", "Gemini-2.5-Pro")

    # Load data for both models
    with open(file1, "r") as f:
        stats_data1 = json.load(f)

    with open(file2, "r") as f:
        stats_data2 = json.load(f)

    # Extract category data for both models
    category_data1 = stats_data1["by_category"]
    category_data2 = stats_data2["by_category"]

    # Find common categories and get top categories by average total count
    combined_categories = {}
    for code in set(category_data1.keys()) & set(category_data2.keys()):
        # Skip cs.IT category as in original code
        if code == "cs.IT":
            continue
        total1 = category_data1[code]["total"]
        total2 = category_data2[code]["total"]
        combined_categories[code] = (total1 + total2) / 2  # Average count

    # Get top categories by average occurrence
    top_categories = sorted(combined_categories.items(), key=lambda x: x[1], reverse=True)[:14]
    category_codes = [cat[0] for cat in top_categories]

    # Get accuracies for both models (convert to percentages)
    accuracies1 = [category_data1[code]["accuracy"] * 100 if code in category_data1 else 0 for code in category_codes]
    accuracies2 = [category_data2[code]["accuracy"] * 100 if code in category_data2 else 0 for code in category_codes]

    # Format category names for display - just use the category code
    category_names = category_codes

    # Set up the figure
    plt.figure(figsize=(12, 8))

    # Vertical bar chart parameters - exchange x and y
    bar_width = 0.35
    x_pos = np.arange(len(category_names))  # X positions for bars

    # Define colors that match the style in other functions
    color1 = '#3498db'  # Blue color used in other plots
    color2 = '#e74c3c'  # Red color used in other plots

    # Plot bars for both models side by side
    bars1 = plt.bar(x_pos - bar_width/2, accuracies1, bar_width, 
                   color=color1, edgecolor='black', linewidth=1.2, label=model1_name)
    bars2 = plt.bar(x_pos + bar_width/2, accuracies2, bar_width, 
                   color=color2, edgecolor='black', linewidth=1.2, label=model2_name)

    # Add percentage labels on top of bars for model 1
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                f'{height:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=12, color='#2980b9')

    # Add percentage labels on top of bars for model 2
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                f'{height:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=12, color='#c0392b')

    # Set labels
    plt.xlabel('Category', fontweight='bold', fontsize=18)
    plt.ylabel('Accuracy (%)', fontweight='bold', fontsize=18)

    # Set x-ticks at bar positions
    plt.xticks(x_pos, category_names, fontsize=14, rotation=45, ha='right')

    # Set y-axis limit with some padding
    plt.ylim(0, max(max(accuracies1), max(accuracies2)) * 1.15)

    # Add grid for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add legend with larger font and position it below the figure
    plt.legend(fontsize=16, frameon=True, loc='upper center', ncol=2)
    # Remove top and right spines for cleaner look
    sns.despine()

    # Adjust layout and expand margins to ensure text fits
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.22)  # Space for category labels and legend

    # Save the plot
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "two_model_comparison_by_category.pdf"), 
                bbox_inches='tight', format='pdf', dpi=300)

    print(f"Comparison plots saved to {plots_dir}")
    plt.close()


def _model_accuracy(subset, output_path=None):
    """tee
    plot model accuracies from JSON files and optionally save the plot.
    
    Parameters:
    -----------
    input_dir : stros.kill
        Path to directory containing JSON files with model results
    output_path : str, optional
        Path where the accuracy plot will be saved. If None, the plot is just returned.
        
    Returns:
    --------
    a dict of model name and accuracy
    """
    # Get all jsonl files in the input directory
    if subset == "math_stackexchange":
        jsonl_files = glob.glob(os.path.join(SUBSET[subset], '*_wo_context_1run.jsonl'))
    else:
        jsonl_files = glob.glob(os.path.join(SUBSET[subset], '*_w_context_1run.jsonl'))
    print(f"Found {len(jsonl_files)} JSONL files")
    
    # Store results
    models_data = []
    
    # Process each file
    for file_path in jsonl_files:
        try:
            # Extract model name from file name
            if subset == "math_stackexchange":
                model_name = os.path.basename(file_path).replace('_wo_context_1run.jsonl', '')
            else:
                model_name = os.path.basename(file_path).replace('_w_context_1run.jsonl', '')
            
            # Load the data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.loads(f.read())
            
            # Extract key metrics
            model_info = {
                'model': model_name,
                'accuracy': data.get('accuracy', 0),
            }
            
            models_data.append(model_info)
            print(f"Processed {model_name}: Accuracy = {model_info['accuracy']:.2%}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(models_data)
    # sort by accuracy
    df = df.sort_values('accuracy', ascending=False)
    # make it into dict
    df_dict = df.to_dict(orient='records')
    return df_dict


def _get_model_color_map():
    """
    Returns a consistent color mapping for all models found in the results directories.
    """
    # Gather all models from all datasets
    cs_arxiv_results = _model_accuracy("cs_arxiv")
    math_arxiv_results = _model_accuracy("math_arxiv")
    math_stackexchange_results = _model_accuracy("math_stackexchange")
    all_models = set()
    for results in [cs_arxiv_results, math_arxiv_results, math_stackexchange_results]:
        all_models.update([d['model'] for d in results])
    all_models = sorted(all_models)
    palette = sns.color_palette("husl", len(all_models))
    model_color_map = {model: palette[i] for i, model in enumerate(all_models)}
    return model_color_map


def plot_frontier_models():
    """
    Create a beautiful visualization with three subplots showing model performance with a shared legend.
    """

    # Load data from files
    cs_arxiv_results = _model_accuracy("cs_arxiv")
    math_arxiv_results = _model_accuracy("math_arxiv")
    math_stackexchange_results = _model_accuracy("math_stackexchange")
    
    # Create dataframes for plotting
    df_cs = pd.DataFrame(cs_arxiv_results)
    df_math = pd.DataFrame(math_arxiv_results)
    df_stack = pd.DataFrame(math_stackexchange_results)
    
    # Convert accuracy from 0-1 to 0-100 for percentages
    df_cs['accuracy'] = df_cs['accuracy'] * 100
    df_math['accuracy'] = df_math['accuracy'] * 100
    df_stack['accuracy'] = df_stack['accuracy'] * 100
    
    # for each dataset, print the model and accuracy
    for df, dataset in zip([df_cs, df_math, df_stack], ["CS ArXiv", "Math ArXiv", "Math StackExchange"]):
        print(f"{dataset} models and their accuracies:")
        for model, accuracy in zip(df['model'], df['accuracy']):
            print(f"{model}: {accuracy:.1f}%")
        print("\n")
        
    
    # Get consistent color mapping
    model_color_map = _get_model_color_map()
    
    # Define specific models to highlight with hatches
    special_models = ['gpt-4o-mini', 'llama-3.1-405b', 'gemini-2.5-pro', 'grok-3']
    
    # Define distinctive hatches for the special models - use 1-to-1 mapping
    model_hatch_map = {
        'gpt-4o-mini': '|',
        'llama-3.1-405b': '\\',
        'gemini-2.5-pro': '/',
        'grok-3': 'x'
    }
    
    # Create a figure with 3 subplots, make the figure full width
    fig, axs = plt.subplots(3, 1, figsize=(15, 8))
    
    # List of datasets and their dataframes
    datasets = [
        ("CS ArXiv", df_cs, axs[0]),
        ("Math ArXiv", df_math, axs[1]),
        ("Math StackExchange", df_stack, axs[2])
    ]
    
    # For legend
    all_handles = []
    all_labels = []
    
    # Plot each dataset
    index = 0
    for title, df, ax in datasets:
        # Create the bar plot - sort by accuracy for better visualization
        df_sorted = df.sort_values('accuracy', ascending=False)
        
        # Create a list of colors that match the sorted model order
        colors = [model_color_map[model] for model in df_sorted['model']]
        
        # Create the plot with specific colors
        bars = sns.barplot(x=np.arange(len(df_sorted)), y='accuracy', data=df_sorted, ax=ax, palette=colors)
        
        # Add hatches only to specific models
        for i, bar in enumerate(bars.patches):
            model_name = df_sorted['model'].iloc[i]
            if model_name in special_models:
                bar.set_hatch(model_hatch_map[model_name])
            
            # Add edge to all bars
            bar.set_edgecolor('black')
            bar.set_linewidth(0.8)
        
        # Set titles and labels
        ax.set_xlabel(title, fontweight='bold', fontsize=20)
        if index == 1:
            ax.set_ylabel('Accuracy (%)', fontsize=20)
        else:
            ax.set_ylabel('')
        
        # Replace model names with numeric indices
        ax.set_xticks(np.arange(len(df_sorted)))
        ax.set_xticklabels([])  # Remove xtick labels
        
        # Set y-axis to start at 0 and end at a reasonable upper bound
        ax.set_ylim(0, 100)
        # Add value labels on top of each bar
        for i, v in enumerate(df_sorted['accuracy']):
            ax.text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontsize=18)
        
        # Add gridlines for easier reading
        ax.grid(axis='y', alpha=0.3)
        
        # Remove top and right spines for cleaner look
        sns.despine(ax=ax)
        
        # Create handles for legend (only for the first subplot)
        if title == "CS ArXiv":
            # Get all unique models across all datasets
            all_models = set(df_cs['model']) | set(df_math['model']) | set(df_stack['model'])
            sorted_models = sorted(all_models)
            
            for model in sorted_models:
                # Get the color for this model
                if model in df_sorted['model'].values:
                    color = model_color_map[model]
                else:
                    # If model not in this dataset, still get its color from the map
                    color = model_color_map[model]
                
                # Determine if this model gets a hatch
                hatch = model_hatch_map.get(model, '')
                
                # Create patch
                patch = plt.Rectangle((0, 0), 1, 1, 
                                     facecolor=color, 
                                     hatch=hatch, 
                                     edgecolor='black',
                                     linewidth=1.5)
                all_handles.append(patch)
                all_labels.append(model)
        index += 1
    
    # Create a separate figure legend with much larger patches to show the hatches clearly
    legend = fig.legend(
        all_handles, 
        all_labels, 
        loc='upper center',  # Position at upper center
        bbox_to_anchor=(0.5, 0.0),  # Move to the very bottom
        ncol=5,  # Use 5 columns
        frameon=True, 
        prop={'size': 13, 'weight': 'bold'},  # Use prop dict for font properties
        handlelength=4,   # Make legend symbols longer
        handleheight=2.5, # Make legend symbols taller
    )
    
    # Set the legend patch alpha to 1.0 to make them fully opaque
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    
    # Set tight layout to remove excess space
    plt.tight_layout()
    
    # Save the figure
    plt.savefig("frontier_models.pdf", dpi=300, bbox_inches='tight')
    return df_sorted


def plot_fine_tuning_models():
    def _model_accuracy_ft(subset, ft=True, run=0):
        """tee
        plot model accuracies from JSON files and optionally save the plot.
        
        Parameters:
        -----------
        input_dir : stros.kill
            Path to directory containing JSON files with model results
        output_path : str, optional
            Path where the accuracy plot will be saved. If None, the plot is just returned.
            
        Returns:
        --------
        a dict of model name and accuracy
        """
        # Get all jsonl files in the input directory
        if ft:
            jsonl_files = glob.glob(os.path.join(SUBSET[subset], f"gpt-4o-mini-ft_w_context_{run+1}run.jsonl"))
        else:
            jsonl_files = glob.glob(os.path.join(SUBSET[subset], f"gpt-4o-mini_w_context_{run+1}run.jsonl"))
        # import pdb; pdb.set_trace()
        print(f"Found {len(jsonl_files)} JSONL files")
        
        # Store results
        models_data = []
        
        # Process each file
        for file_path in jsonl_files:
            try:
                model_name = os.path.basename(file_path).replace('_w_context_{}run.jsonl'.format(run), '')
                
                # Load the data
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.loads(f.read())
                
                # Extract key metrics
                model_info = {
                    'model': model_name,
                    'accuracy': data.get('accuracy', 0),
                }
                
                models_data.append(model_info)
                print(f"Processed {model_name}: Accuracy = {model_info['accuracy']:.2%}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(models_data)
        
        # sort by accuracy
        df = df.sort_values('accuracy', ascending=False)
        # make it into dict
        df_dict = df.to_dict(orient='records')
        return df_dict[0] if df_dict else None

    # 5 different runs
    ft_results = []
    no_ft_results = []
    for run in range(5):
        # load the results
        results_ft = _model_accuracy_ft("math_arxiv_ft", ft=True, run=run)
        results_no_ft = _model_accuracy_ft("math_arxiv_ft", ft=False, run=run)
        if results_ft and results_no_ft:
            ft_results.append(results_ft['accuracy'] * 100)  # Convert to percentage
            no_ft_results.append(results_no_ft['accuracy'] * 100)  # Convert to percentage

    # Calculate means and standard deviations
    ft_mean = np.mean(ft_results)
    ft_std = np.std(ft_results)
    no_ft_mean = np.mean(no_ft_results)
    no_ft_std = np.std(no_ft_results)

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Set up the bar positions
    x = np.arange(2)
    width = 0.35
    
    # Create bars with error bars
    bars = ax.bar(x, [no_ft_mean, ft_mean], width, 
                 yerr=[no_ft_std, ft_std],
                 capsize=10,
                 color=['#3498db', '#e74c3c'],
                 edgecolor='black',
                 linewidth=1.5)
    
    # Customize the plot
    ax.set_ylabel('Accuracy (%)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(['Original', 'Fine-tuned'], fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7, axis='y')
    ax.set_ylim(0, 14)
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig("fine_tuning_comparison.pdf", dpi=300, bbox_inches='tight')


def plot_model_release_date_math_arxiv_hardest():
    # Load data from files
    def _model_accuracy_on_hardest(subset, output_path=None):
        """tee
        plot model accuracies from JSON files and optionally save the plot.
        
        Parameters:
        -----------
        input_dir : stros.kill
            Path to directory containing JSON files with model results
        output_path : str, optional
            Path where the accuracy plot will be saved. If None, the plot is just returned.
            
        Returns:
        --------
        a dict of model name and accuracy
        """
        jsonl_files = glob.glob(os.path.join(SUBSET[subset], '*results_with_regrouped_difficulty.json'))
        print(f"Found {len(jsonl_files)} JSONL files")
        
        # Store results
        models_data = []
        
        # Process each file
        for file_path in jsonl_files:
            try:

                model_name = os.path.basename(file_path).replace('_results_with_regrouped_difficulty.json', '')
                
                # Load the data
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.loads(f.read())
                
                # Extract key metrics
                model_info = {
                    'model': model_name,
                    'accuracy': data.get('accuracy_g3', 0),
                }
                
                models_data.append(model_info)
                print(f"Processed {model_name}: Accuracy = {model_info['accuracy']:.2%}")
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(models_data)
        # sort by accuracy
        df = df.sort_values('accuracy', ascending=False)
        # make it into dict
        df_dict = df.to_dict(orient='records')
        return df_dict


    math_arxiv_results = _model_accuracy_on_hardest("math_arxiv_hardest")

    # Create dataframes
    df_math = pd.DataFrame(math_arxiv_results)

    # skip qwen
    # import pdb; pdb.set_trace()
    df_math = df_math[df_math['model'] != 'qwen3-235b']
    # Convert accuracy to percentages
    df_math['accuracy'] = df_math['accuracy'] * 100

    # Add dataset labels
    df_math['dataset'] = 'Math ArXiv'

    df_combined = df_math
    # Add release dates
    release_dates = MODEL_RELEASE_DATE
    df_combined['release_date'] = df_combined['model'].map(release_dates)
    df_combined['release_date'] = pd.to_datetime(df_combined['release_date'])

    # Compute average accuracy
    model_avg = df_combined.groupby(['model', 'release_date'])['accuracy'].mean().reset_index()
    model_avg = model_avg.sort_values('release_date')

    # Use consistent color mapping
    model_color_map = _get_model_color_map()

    # Plot setup
    fig, ax = plt.subplots(figsize=(15, 8))
    markers = ['1', 's', 'D', 'P', 'X', 'o', 'p', 'h', 'v', '^', '<', '>']
    # make the '*' marker larger in the legend
    
    handles = []
    texts = []

    # Use only unique release dates as x-ticks (categorical axis)
    unique_dates = sorted(model_avg['release_date'].unique())
    xtick_labels = [d.strftime('%b %Y') for d in unique_dates]

    for i, (idx, row) in enumerate(model_avg.iterrows()):
        marker_idx = i % len(markers)
        # Use categorical x-position
        x_pos = unique_dates.index(row['release_date'])
        sc = ax.scatter(
            x_pos,
            row['accuracy'],
            s=550,
            marker=markers[marker_idx],
            color=model_color_map[row['model']],
            edgecolors='white',
            linewidths=2,
            zorder=3,
            label=row['model']
        )
        # Manual offset for 52.2% and 54.6% labels
        if np.isclose(row['accuracy'], 27.9, atol=0.05):
            # move the label to the left
            x_pos = x_pos - 0.5
            y_offset = -1
            va = 'bottom'
        elif np.isclose(row['accuracy'], 49.1, atol=0.05):
            # move the label to the right
            x_pos = x_pos + 0.5
            y_offset = 2.5
            va = 'bottom'
        else:
            y_offset = 2.5
            va = 'bottom'
        texts.append(
            ax.text(
                x_pos,
                row['accuracy'] + y_offset,
                f"{row['accuracy']:.1f}%",
                ha='center',
                va=va,
                fontsize=15,
                fontweight='bold',
                color='#222',
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.85, lw=1.2),
                zorder=10
            )
        )
        handles.append(sc)

    # Use adjustText to avoid overlap and add arrows
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(
            arrowstyle="->",
            color='#888',
            lw=1.2,
            alpha=0.7,
            shrinkA=15,
            shrinkB=5
        ),
        expand_points=(2, 2),
        expand_text=(2, 2),
        force_text=2,
        force_points=1,
        only_move={'points':'xy', 'text':'xy'}
    )

    # Set categorical x-ticks
    ax.set_xticks(range(len(unique_dates)))
    ax.set_xticklabels(xtick_labels, rotation=30, ha='right', fontsize=16)

    # Y-axis and grid
    ax.set_ylabel('Accuracy (%)', fontsize=20, labelpad=10)
    ax.set_xlabel('Release Date', fontsize=20, labelpad=10, fontweight='bold')
    ax.set_ylim(-2, model_avg['accuracy'].max() + 4)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    sns.despine(ax=ax)

    # Add legend for model names, sorted by accuracy
    # update the model names to uppercase
    model_avg['model'] = model_avg['model'].map(UPPERCASE_MODELS)
    legend_data = []
    for h, l in zip(handles, model_avg['model']):
        if l not in [x[1] for x in legend_data]:
            # Find the accuracy for this model
            acc = model_avg[model_avg['model'] == l]['accuracy'].values[0]
            legend_data.append((h, l, acc))
    # Sort by accuracy descending
    legend_data.sort(key=lambda x: -x[2])
    handles_unique = [x[0] for x in legend_data]
    labels_unique = [x[1] for x in legend_data]

    ax.legend(
        handles_unique,
        labels_unique,
        title='Models',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=18,
        title_fontsize=18,
        frameon=True,
        labelspacing=1.5
    )

    plt.savefig("teaser.png", dpi=300, bbox_inches='tight')
    plt.close()



def plot_model_release_date_math_arxiv():
    # Load data from files
    cs_arxiv_results = _model_accuracy("cs_arxiv")
    math_arxiv_results = _model_accuracy("math_arxiv")
    math_stackexchange_results = _model_accuracy("math_stackexchange")

    # Create dataframes
    df_cs = pd.DataFrame(cs_arxiv_results)
    df_math = pd.DataFrame(math_arxiv_results)
    df_stack = pd.DataFrame(math_stackexchange_results)

    # Convert accuracy to percentages
    df_cs['accuracy'] = df_cs['accuracy'] * 100
    df_math['accuracy'] = df_math['accuracy'] * 100
    df_stack['accuracy'] = df_stack['accuracy'] * 100

    # Add dataset labels
    df_cs['dataset'] = 'CS ArXiv'
    df_math['dataset'] = 'Math ArXiv'
    df_stack['dataset'] = 'Math StackExchange'

    # Combine datasets
    # df_combined = pd.concat([df_cs, df_math, df_stack])
    df_combined = df_math
    # Add release dates
    release_dates = MODEL_RELEASE_DATE
    df_combined['release_date'] = df_combined['model'].map(release_dates)
    df_combined['release_date'] = pd.to_datetime(df_combined['release_date'])

    # Compute average accuracy
    model_avg = df_combined.groupby(['model', 'release_date'])['accuracy'].mean().reset_index()
    model_avg = model_avg.sort_values('release_date')

    # Use consistent color mapping
    model_color_map = _get_model_color_map()

    # Plot setup
    fig, ax = plt.subplots(figsize=(15, 8))
    markers = ['1', 's', 'D', 'P', 'X', 'o', 'p', 'h', 'v', '^', '<', '>']
    # make the '*' marker larger in the legend
    
    handles = []
    texts = []

    # Use only unique release dates as x-ticks (categorical axis)
    unique_dates = sorted(model_avg['release_date'].unique())
    xtick_labels = [d.strftime('%b %Y') for d in unique_dates]

    for i, (idx, row) in enumerate(model_avg.iterrows()):
        marker_idx = i % len(markers)
        # Use categorical x-position
        x_pos = unique_dates.index(row['release_date'])
        sc = ax.scatter(
            x_pos,
            row['accuracy'],
            s=550,
            marker=markers[marker_idx],
            color=model_color_map[row['model']],
            edgecolors='white',
            linewidths=2,
            zorder=3,
            label=row['model']
        )
        # Manual offset for 52.2% and 54.6% labels
        if np.isclose(row['accuracy'], 43.4, atol=0.05):
            # move the label to the left
            x_pos = x_pos - 0.5
            y_offset = -2.
            va = 'bottom'
        elif np.isclose(row['accuracy'], 49.1, atol=0.05):
            # move the label to the right
            x_pos = x_pos + 0.5
            y_offset = 2.5
            va = 'bottom'
        else:
            y_offset = 2.5
            va = 'bottom'
        texts.append(
            ax.text(
                x_pos,
                row['accuracy'] + y_offset,
                f"{row['accuracy']:.1f}%",
                ha='center',
                va=va,
                fontsize=15,
                fontweight='bold',
                color='#222',
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.85, lw=1.2),
                zorder=10
            )
        )
        handles.append(sc)

    # Use adjustText to avoid overlap and add arrows
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(
            arrowstyle="->",
            color='#888',
            lw=1.2,
            alpha=0.7,
            shrinkA=15,
            shrinkB=5
        ),
        expand_points=(2, 2),
        expand_text=(2, 2),
        force_text=2,
        force_points=1,
        only_move={'points':'xy', 'text':'xy'}
    )

    # Set categorical x-ticks
    ax.set_xticks(range(len(unique_dates)))
    ax.set_xticklabels(xtick_labels, rotation=30, ha='right', fontsize=16)

    # Y-axis and grid
    ax.set_ylabel('Accuracy (%)', fontsize=20, labelpad=10)
    ax.set_xlabel('Release Date', fontsize=20, labelpad=10, fontweight='bold')
    ax.set_ylim(10, model_avg['accuracy'].max() + 4)
    ax.grid(True, linestyle='--', alpha=0.4, zorder=0)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    sns.despine(ax=ax)

    # Add legend for model names, sorted by accuracy
    # update the model names to uppercase
    model_avg['model'] = model_avg['model'].map(UPPERCASE_MODELS)
    legend_data = []
    for h, l in zip(handles, model_avg['model']):
        if l not in [x[1] for x in legend_data]:
            # Find the accuracy for this model
            acc = model_avg[model_avg['model'] == l]['accuracy'].values[0]
            legend_data.append((h, l, acc))
    # Sort by accuracy descending
    legend_data.sort(key=lambda x: -x[2])
    handles_unique = [x[0] for x in legend_data]
    labels_unique = [x[1] for x in legend_data]

    ax.legend(
        handles_unique,
        labels_unique,
        title='Models',
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        fontsize=18,
        title_fontsize=18,
        frameon=True,
        labelspacing=1.5
    )

    plt.savefig("teaser.png", dpi=300, bbox_inches='tight')
    plt.close()




def plot_cutoff():
    # List of models to analyze
    models = [
        ('gpt-4o-mini', 'gpt-4o-mini_w_context_1run.jsonl'),
        ('llama-3.1-405b', 'llama-3.1-405b_w_context_1run.jsonl'),
        ('claude-3.5-sonnet', 'claude-3.5-sonnet_w_context_1run.jsonl'),
    ]
    cutoff_date = "2024-04-01"
    cutoff_year = int(cutoff_date.split('-')[0])
    cutoff_month = int(cutoff_date.split('-')[1])

    # Get data for all models
    all_models_data = []
    
    for model_name, json_filename in models:
        json_file = os.path.join(SUBSET["math_arxiv"], json_filename)
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
        results = data['results']
        before_cutoff = []
        after_cutoff = []
        for result in results:
            paper_id = result['paper_link'].split('/')[-1]
            if 'v' in paper_id:
                paper_id = paper_id.split('v')[0]
            if len(paper_id) >= 4 and paper_id[:4].isdigit():
                yy = int(paper_id[:2])
                mm = int(paper_id[2:4])
                year = 2000 + yy
                month = mm
                if year < cutoff_year:
                    before_cutoff.append(result)
                elif year == cutoff_year and month <= cutoff_month:
                    before_cutoff.append(result)
                else:
                    after_cutoff.append(result)
                    
        if before_cutoff:
            before_acc = sum([r['is_correct'] for r in before_cutoff]) / len(before_cutoff)
        else:
            before_acc = 0
        if after_cutoff:
            after_acc = sum([r['is_correct'] for r in after_cutoff]) / len(after_cutoff)
        else:
            after_acc = 0
            
        all_models_data.append({
            'model': UPPERCASE_MODELS[model_name],
            'before_acc': before_acc * 100,
            'after_acc': after_acc * 100,
            'before_count': len(before_cutoff),
            'after_count': len(after_cutoff)
        })
    
    # Create figure for combined plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set width of bars and positions
    bar_width = 0.35
    model_positions = np.arange(len(all_models_data))
    
    # Plot bars for before and after cutoff
    before_bars = ax.bar(
        model_positions - bar_width/2, 
        [d['before_acc'] for d in all_models_data], 
        bar_width, 
        label='Before Cutoff', 
        color='#3498db'
    )
    
    after_bars = ax.bar(
        model_positions + bar_width/2, 
        [d['after_acc'] for d in all_models_data], 
        bar_width, 
        label='After Cutoff', 
        color='#e74c3c'
    )
    
    # Add labels and title
    ax.set_ylabel('Accuracy (%)', fontsize=20)
    ax.set_xticks(model_positions)
    ax.set_xticklabels([d['model'] for d in all_models_data], fontsize=20)
    ax.set_ylim(0, 30)
    ax.legend(fontsize=20)
    
    # Add value labels on top of bars with counts
    for i, d in enumerate(all_models_data):
        if d['before_count'] > 0:
            ax.text(
                i - bar_width/2, 
                d['before_acc'] + 1, 
                f"{d['before_acc']:.1f}%\n(n={d['before_count']})", 
                ha='center', 
                va='bottom', 
                fontsize=20
            )
        if d['after_count'] > 0:
            ax.text(
                i + bar_width/2, 
                d['after_acc'] + 1, 
                f"{d['after_acc']:.1f}%\n(n={d['after_count']})", 
                ha='center', 
                va='bottom', 
                fontsize=20
            )
    
    # Add a title explaining the cutoff date
    # plt.title(f"Model Performance Before and After Cutoff Date ({cutoff_date})", fontsize=18, fontweight='bold')
    
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig("cutoff_combined.pdf", dpi=300, bbox_inches='tight')
    plt.show()


def visualize_error_categories():
    """
    Create a visualization comparing error categories across multiple models.
    """
    model_data_paths = [
        "/data/projects/math_bench/analyze/o3_categorized_errors_filtered_summary.json",
        "/data/projects/math_bench/analyze/gemini-categorized_errors_filtered_summary.json",
        "/data/projects/math_bench/analyze/r1-categorized_errors_filtered_summary.json"
    ]
    
    # Model names 
    model_ids = ["o3", "gemini-2.5-pro", "deepseek-r1"]
    
    # Get display names using UPPERCASE_MODELS
    model_names = [UPPERCASE_MODELS.get(model_id, model_id) for model_id in model_ids]
    output_prefix = 'error_categories'

    # Use the same colors as plot_cutoff function plus one more
    # Blue, red, and green - the same colors used in other plots
    model_colors = ['#3498db', '#e74c3c', '#2ecc71']

    # Load data for all models
    model_categories = []
    for data_path in model_data_paths:
        with open(data_path, "r") as f:
            error_data = json.load(f)
            
        # Process data
        categories = {}
        for category, percentage in error_data["percentage_distribution"].items():
            category_text = category.replace('_', ' ').title()
            if category_text == "Unclear Or Incomplete Answer":
                category_text = "Incomplete or Unclear Answer"
            categories[category_text] = percentage
            
        model_categories.append(categories)
    
    # Combine all unique categories
    all_categories = sorted(set().union(*[categories.keys() for categories in model_categories]))
    
    # Function to split category names more effectively
    def split_category_name(name):
        # Special case for "Incomplete or Unclear Answer"
        if name == "Incomplete or Unclear Answer":
            return "Incomplete or\nUnclear Answer"
        
        words = name.split()
        
        # For long category names (4+ words), use multiple lines
        if len(words) >= 4:
            third = len(words) // 3
            return ' '.join(words[:third]) + '\n' + ' '.join(words[third:2*third]) + '\n' + ' '.join(words[2*third:])
        
        # For medium category names, use 2 lines
        elif len(words) >= 2:
            mid = len(words) // 2
            return ' '.join(words[:mid]) + '\n' + ' '.join(words[mid:])
        
        return name
    
    # Set up data for plotting
    formatted_categories = [split_category_name(category) for category in all_categories]
    model_percentages = []
    
    for categories in model_categories:
        percentages = [categories.get(category, 0) for category in all_categories]
        model_percentages.append(percentages)
    
    # Sort based on the average percentage (descending)
    avg_percentages = [sum(percentages) / len(model_percentages) for percentages in zip(*model_percentages)]
    combined = sorted(zip(formatted_categories, *model_percentages), 
                      key=lambda x: avg_percentages[formatted_categories.index(x[0])], 
                      reverse=True)
    
    # Unpack the sorted data
    sorted_data = list(zip(*combined))
    formatted_categories = sorted_data[0]
    model_percentages = sorted_data[1:]
    
    # Create a figure with appropriate dimensions
    plt.figure(figsize=(16, 8))
    
    # Set the width of the bars and their positions
    bar_width = 0.25
    index = np.arange(len(formatted_categories))
    
    # Create the grouped bar chart
    bars = []
    for i, (percentages, model_name) in enumerate(zip(model_percentages, model_names)):
        position = index - bar_width + (i + 0.5) * bar_width
        
        # Get color from model_colors list
        color = model_colors[i % len(model_colors)]
        
        bar = plt.bar(position, percentages, bar_width, 
                      color=color, 
                      edgecolor='black', 
                      linewidth=1.0,
                      label=model_name)
        bars.append(bar)
    
    # Customize the plot
    plt.ylabel('Errors (%)', fontsize=20)
    max_percentage = max([max(percentages) for percentages in model_percentages])
    plt.ylim(0, max_percentage * 1.15)
    plt.xticks(index, formatted_categories, rotation=0, ha='center', fontsize=16)
    
    # Create a legend with larger font size and position it below the figure
    plt.legend(fontsize=20, frameon=True, bbox_to_anchor=(0.5, -0.15), loc='upper center', ncol=3)
    
    # Add grid for consistency with other plots
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Remove the top and right spines for cleaner look
    sns.despine()
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)  # Extra space at the bottom for the legend
    
    # Save the plot in both PDF and PNG formats
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f"{output_prefix}.pdf"), 
                bbox_inches='tight', format='pdf', dpi=300)
    
    print(f"Error categories plot saved to {plots_dir}")
    plt.close()


def plot_combined_category_cutoff():
    """
    Creates a combined figure with plot_per_category_math_arxiv (60% width) on the left
    and plot_cutoff (40% width) on the right.
    """
    # Create a single figure with two subplots of specified width ratios
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[6, 4])
    
    # Left subplot (60%) - Category comparison
    ax1 = fig.add_subplot(gs[0])
    
    # Define file paths for both models
    file1 = "/data/projects/math_bench/analyze/category_statistics_o3-results_annotated_math_level_arxiv_math_math_categories.json"
    file2 = "/data/projects/math_bench/analyze/category_statistics_gemini-2_annotated_math_level_arxiv_math_math_categories.json"

    # Extract model names from filenames for the legend
    model1_name = UPPERCASE_MODELS.get("o3", "o3") 
    model2_name = UPPERCASE_MODELS.get("gemini-2.5-pro", "Gemini-2.5-Pro")

    # Load data for both models
    with open(file1, "r") as f:
        stats_data1 = json.load(f)

    with open(file2, "r") as f:
        stats_data2 = json.load(f)

    # Extract category data for both models
    category_data1 = stats_data1["by_category"]
    category_data2 = stats_data2["by_category"]

    # Find common categories and get top categories by average total count
    combined_categories = {}
    for code in set(category_data1.keys()) & set(category_data2.keys()):
        # Skip cs.IT category as in original code
        if code == "cs.IT":
            continue
        total1 = category_data1[code]["total"]
        total2 = category_data2[code]["total"]
        combined_categories[code] = (total1 + total2) / 2  # Average count

    # Get top categories by average occurrence
    top_categories = sorted(combined_categories.items(), key=lambda x: x[1], reverse=True)[:14]
    category_codes = [cat[0] for cat in top_categories]

    # Get accuracies for both models (convert to percentages)
    accuracies1 = [category_data1[code]["accuracy"] * 100 if code in category_data1 else 0 for code in category_codes]
    accuracies2 = [category_data2[code]["accuracy"] * 100 if code in category_data2 else 0 for code in category_codes]

    # Format category names for display - just use the category code
    category_names = category_codes

    # Vertical bar chart parameters - exchange x and y
    bar_width = 0.35
    x_pos = np.arange(len(category_names))  # X positions for bars

    # Define colors that match the style in other functions
    color1 = '#3498db'  # Blue color used in other plots
    color2 = '#e74c3c'  # Red color used in other plots

    # Plot bars for both models side by side
    bars1 = ax1.bar(x_pos - bar_width/2, accuracies1, bar_width, 
                   color=color1, edgecolor='black', linewidth=1.2, label=model1_name)
    bars2 = ax1.bar(x_pos + bar_width/2, accuracies2, bar_width, 
                   color=color2, edgecolor='black', linewidth=1.2, label=model2_name)

    # Add percentage labels on top of bars for model 1
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                f'{height:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=12, color='#2980b9')

    # Add percentage labels on top of bars for model 2
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.5, 
                f'{height:.1f}%', ha='center', va='bottom', 
                fontweight='bold', fontsize=12, color='#c0392b')

    # Set labels
    # ax1.set_xlabel('Category', fontweight='bold', fontsize=18)
    ax1.set_ylabel('Accuracy (%)', fontsize=20)

    # Set x-ticks at bar positions
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(category_names, fontsize=14, rotation=45, ha='right')

    # Set y-axis limit with some padding
    ax1.set_ylim(0, max(max(accuracies1), max(accuracies2)) * 1.15)

    # Add grid for better readability
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Add legend with larger font and position it below the figure
    ax1.legend(fontsize=20, frameon=True, loc='upper center', ncol=2)
    
    # Remove top and right spines for cleaner look
    sns.despine(ax=ax1)

    # Right subplot (40%) - Cutoff comparison
    ax2 = fig.add_subplot(gs[1])
    
    # List of models to analyze
    models = [
        ('gpt-4o-mini', 'gpt-4o-mini_w_context_1run.jsonl'),
        ('llama-3.1-405b', 'llama-3.1-405b_w_context_1run.jsonl'),
        ('claude-3.5-sonnet', 'claude-3.5-sonnet_w_context_1run.jsonl'),
    ]
    cutoff_date = "2024-04-01"
    cutoff_year = int(cutoff_date.split('-')[0])
    cutoff_month = int(cutoff_date.split('-')[1])

    # Get data for all models
    all_models_data = []
    
    for model_name, json_filename in models:
        json_file = os.path.join(SUBSET["math_arxiv"], json_filename)
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.loads(f.read())
        results = data['results']
        before_cutoff = []
        after_cutoff = []
        for result in results:
            paper_id = result['paper_link'].split('/')[-1]
            if 'v' in paper_id:
                paper_id = paper_id.split('v')[0]
            if len(paper_id) >= 4 and paper_id[:4].isdigit():
                yy = int(paper_id[:2])
                mm = int(paper_id[2:4])
                year = 2000 + yy
                month = mm
                if year < cutoff_year:
                    before_cutoff.append(result)
                elif year == cutoff_year and month <= cutoff_month:
                    before_cutoff.append(result)
                else:
                    after_cutoff.append(result)
                    
        if before_cutoff:
            before_acc = sum([r['is_correct'] for r in before_cutoff]) / len(before_cutoff)
        else:
            before_acc = 0
        if after_cutoff:
            after_acc = sum([r['is_correct'] for r in after_cutoff]) / len(after_cutoff)
        else:
            after_acc = 0
            
        all_models_data.append({
            'model': UPPERCASE_MODELS[model_name],
            'before_acc': before_acc * 100,
            'after_acc': after_acc * 100,
            'before_count': len(before_cutoff),
            'after_count': len(after_cutoff)
        })
    
    # Set width of bars and positions
    bar_width = 0.35
    model_positions = np.arange(len(all_models_data))
    
    # Plot bars for before and after cutoff
    before_bars = ax2.bar(
        model_positions - bar_width/2, 
        [d['before_acc'] for d in all_models_data], 
        bar_width, 
        label='Before Cutoff', 
        color='#3498db'
    )
    
    after_bars = ax2.bar(
        model_positions + bar_width/2, 
        [d['after_acc'] for d in all_models_data], 
        bar_width, 
        label='After Cutoff', 
        color='#e74c3c'
    )
    
    # Add labels and title
    ax2.set_ylabel('Accuracy (%)', fontsize=20)
    ax2.set_xticks(model_positions)
    ax2.set_xticklabels([d['model'] for d in all_models_data], fontsize=18)
    ax2.set_ylim(0, 30)
    ax2.legend(fontsize=20, frameon=True, loc='upper center', ncol=2)
    
    # Add value labels on top of bars with counts
    for i, d in enumerate(all_models_data):
        if d['before_count'] > 0:
            ax2.text(
                i - bar_width/2, 
                d['before_acc'] + 1, 
                f"{d['before_acc']:.1f}%\n(n={d['before_count']})", 
                ha='center', 
                va='bottom', 
                fontsize=16
            )
        if d['after_count'] > 0:
            ax2.text(
                i + bar_width/2, 
                d['after_acc'] + 1, 
                f"{d['after_acc']:.1f}%\n(n={d['after_count']})", 
                ha='center', 
                va='bottom', 
                fontsize=16
            )
    
    # Remove top and right spines for cleaner look
    sns.despine(ax=ax2)
    
    # Adjust layout for both subplots
    plt.tight_layout()
    plt.savefig("combined_category_cutoff.pdf", dpi=300, bbox_inches='tight')
    
    
    plt.close()


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Plot model accuracies on math benchmark.')
    parser.add_argument('--output', type=str, default="model_performance_comparison.png", 
                        help='Path where the accuracy plot will be saved')
    args = parser.parse_args()
    
    # Generate and save the plots using data from files
    # plot_cutoff()
    # plot_frontier_models()
    # plot_model_release_date_math_arxiv()
    # radar_models()  # Uncomment to generate radar chart
    # plot_fine_tuning_models()
    # plot_per_category_math_arxiv()
    # visualize_error_categories()
    # plot_combined_category_cutoff()  # New combined plot
    plot_model_release_date_math_arxiv_hardest()