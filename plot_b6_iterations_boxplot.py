#!/usr/bin/env python3
"""
Generate grouped bar plots or line plots from b6_iterations_table.csv
Aggregates across all datasets, with token_budget on x-axis,
models as groups, and total_iterations on y-axis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

def create_grouped_barplots(csv_path, output_path=None):
    """
    Create grouped bar plots from the B6 iterations data.
    Aggregates across all datasets.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Optional path to save the figure (if None, displays instead)
    """
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    token_budgets = sorted(df['token_budget'].unique())
    models = ['Gemma-3 1B', 'Gemma-3 4B', 'MedGemma 4B', 'Gemma-3 27B', 'MedGemma 27B']
    
    # Prepare data for grouped bar plot
    n_budgets = len(token_budgets)
    n_models = len(models)
    bar_width = 0.15
    x_positions = np.arange(n_budgets)
    
    legend_handles = []
    legend_labels = []
    
    colors = [f'C{i}' for i in range(n_models)]
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        
        means = []
        stds = []
        
        for token_budget in token_budgets:
            subset = model_data[model_data['token_budget'] == token_budget]['total_iterations']
            
            if len(subset) > 0:
                means.append(subset.mean())
                stds.append(subset.std())
            else:
                means.append(0)
                stds.append(0)
        
        # Calculate x position for this model's bars
        x_offset = x_positions + (i - n_models/2 + 0.5) * bar_width
        
        handle = ax.bar(x_offset, means, bar_width, 
                       yerr=stds,
                       label=model,
                       color=colors[i],
                       capsize=4,
                       error_kw={'linewidth': 1.5})
        
        legend_labels.append(model)
        legend_handles.append(handle)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(tb) for tb in token_budgets])
    ax.set_xlabel('Token Budget', fontsize=14)
    ax.set_ylabel('Total Iterations', fontsize=14)
    ax.set_title('PCCR Iterations vs Token Budget (Aggregated Across Datasets)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(legend_labels),
            fontsize=12,
        )
    
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()
    
    return fig

def create_lineplot(csv_path, output_path=None):
    """
    Create line plot from the B6 iterations data.
    Aggregates across all datasets.
    
    Args:
        csv_path: Path to the CSV file
        output_path: Optional path to save the figure (if None, displays instead)
    """
    df = pd.read_csv(csv_path)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    token_budgets = sorted(df['token_budget'].unique())
    models = ['Gemma-3 1B', 'Gemma-3 4B', 'MedGemma 4B', 'Gemma-3 27B', 'MedGemma 27B']
    
    n_models = len(models)
    legend_handles = []
    legend_labels = []
    
    colors = [f'C{i}' for i in range(n_models)]
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        
        means = []
        stds = []
        
        for token_budget in token_budgets:
            subset = model_data[model_data['token_budget'] == token_budget]['total_iterations']
            
            if len(subset) > 0:
                means.append(subset.mean())
                stds.append(subset.std())
            else:
                means.append(0)
                stds.append(0)
        
        handle = ax.errorbar(
            token_budgets,
            means,
            yerr=stds,
            marker='o',
            label=model,
            color=colors[i],
            linestyle='-',
            capsize=3,
            linewidth=2,
            markersize=6
        )
        
        legend_labels.append(model)
        legend_handles.append(handle)
    
    ax.set_xticks(token_budgets)
    ax.set_xticklabels([str(tb) for tb in token_budgets])
    ax.set_xlabel('Token Budget', fontsize=14)
    ax.set_ylabel('Total Iterations', fontsize=14)
    ax.set_title('PCCR Iterations vs Token Budget (Aggregated Across Datasets)', fontsize=16, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='lower center',
            bbox_to_anchor=(0.5, 0.0),
            ncol=len(legend_labels),
            fontsize=12,
        )
    
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {output_path}")
    else:
        plt.show()
    
    return fig

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate grouped bar plots or line plots from B6 iterations data'
    )
    parser.add_argument(
        '--csv',
        type=str,
        default='b6_iterations_table.csv',
        help='Path to the CSV file (default: b6_iterations_table.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for the figure (default: display instead of saving)'
    )
    parser.add_argument(
        '--plot_type',
        type=str,
        choices=['bar', 'line'],
        default='bar',
        help='Type of plot to generate: bar or line (default: bar)'
    )
    
    args = parser.parse_args()
    
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    output_path = args.output
    if output_path and not output_path.endswith(('.png', '.pdf', '.svg', '.jpg')):
        output_path += '.png'
    
    if args.plot_type == 'line':
        create_lineplot(csv_path, output_path)
    else:
        create_grouped_barplots(csv_path, output_path)

if __name__ == '__main__':
    main()
