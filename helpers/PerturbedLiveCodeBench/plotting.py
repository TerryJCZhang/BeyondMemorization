import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_comparison(gpt4_percentages, gpt4o_percentages):
    """
    Plot the accuracy comparison using the extracted arrays.
    
    Args:
        gpt4_percentages: numpy array of GPT-4 accuracies
        gpt4o_percentages: numpy array of GPT-4o accuracies
    """
    # Month labels for x-axis
    months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb']
    x = np.arange(len(months))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot the lines - matching the original style
    ax.plot(x, gpt4_percentages, color='#ff7f0e', linewidth=2.5, label='GPT-4')
    ax.plot(x, gpt4o_percentages, color='#d62728', linewidth=2.5, label='GPT-4o')
    
    # Add vertical dashed lines for knowledge cutoffs
    ax.axvline(x=6, color='#d62728', linestyle='--', linewidth=1.5, alpha=0.7)  # November - GPT-4o color
    ax.axvline(x=7, color='#ff7f0e', linestyle='--', linewidth=1.5, alpha=0.7)  # December - GPT-4 color
    
    # Styling to match the reference image
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, 9.5)
    
    # Set labels
    ax.set_ylabel('Pass@1', fontsize=20)
    ax.set_title('Accuracy on perturbed LiveCodeBench', fontsize=20, pad=20)
    
    # Set x-axis ticks and labels
    ax.set_xticks(x)
    ax.set_xticklabels(months, fontsize=16)
    
    # Set y-axis ticks
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_yticklabels(np.arange(0, 101, 20), fontsize=14)
    
    # Add grid with light style
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=12, frameon=True, fancybox=False, 
              edgecolor='black', framealpha=1, borderpad=1)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('livecodebench_accuracy_plot.png', dpi=150, bbox_inches='tight')
    
    # Display the plot
    plt.show()
    
    # Print the values for reference
    print("\n" + "="*60)
    print("PLOTTED VALUES")
    print("="*60)
    print("GPT-4 percentages:", gpt4_percentages)
    print("GPT-4o percentages:", gpt4o_percentages)
    
    # Calculate and print statistics
    gpt4_valid = gpt4_percentages[gpt4_percentages > 0]
    gpt4o_valid = gpt4o_percentages[gpt4o_percentages > 0]
    
    if len(gpt4_valid) > 0:
        print(f"\nGPT-4 Statistics:")
        print(f"  Average: {np.mean(gpt4_valid):.2f}%")
        print(f"  Min: {np.min(gpt4_valid):.2f}%")
        print(f"  Max: {np.max(gpt4_valid):.2f}%")
    
    if len(gpt4o_valid) > 0:
        print(f"\nGPT-4o Statistics:")
        print(f"  Average: {np.mean(gpt4o_valid):.2f}%")
        print(f"  Min: {np.min(gpt4o_valid):.2f}%")
        print(f"  Max: {np.max(gpt4o_valid):.2f}%")
    
    print("="*60)
    
    return fig, ax


if __name__ == "__main__":
    # Check if arrays exist in the current namespace
    try:
        # These variables should be set by running the analysis code first
        print("Using accuracy arrays from analysis...")
        fig, ax = plot_accuracy_comparison(gpt4_percentages, gpt4o_percentages)
        print("\n✓ Plot saved as 'livecodebench_accuracy_plot.png'")
        
    except NameError:
        print("ERROR: Accuracy arrays not found!")
        print("Please run the JSON analysis code first to extract the arrays.")
        print("\nAlternatively, you can manually set the arrays like this:")
        print("  gpt4_percentages = np.array([...])")
        print("  gpt4o_percentages = np.array([...])")
        print("Then run: plot_accuracy_comparison(gpt4_percentages, gpt4o_percentages)")
