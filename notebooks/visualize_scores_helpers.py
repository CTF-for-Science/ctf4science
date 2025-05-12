# # Visualization Google Colab Of Self-Reported Scores
# @Author: Philippe Wyder  
# (Modified by Alexey Yermakov)

# ## Imports / Flags

import os
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import matplotlib.font_manager as fm
from plotly.subplots import make_subplots
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_theme()

# Global plotting parameters - Updated for NeurIPS requirements
# If Open Sans is not available, use a fallback
font_family = ["Open Sans"]
print(f"Using font family: {font_family}")
plt.rcParams["font.family"] = font_family
plt.rcParams["font.size"] = 8           # General text size (default)
plt.rcParams["axes.labelsize"] = 8      # Axis labels
plt.rcParams["axes.titlesize"] = 9      # Subplot titles
plt.rcParams["legend.fontsize"] = 8     # Legend
plt.rcParams["xtick.labelsize"] = 8     # Tick labels
plt.rcParams["ytick.labelsize"] = 8
plt.rcParams["figure.titlesize"] = 10   # Figure titles

# Line weights
plt.rcParams["lines.linewidth"] = 1.0
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["grid.linewidth"] = 0.5
plt.rcParams["patch.linewidth"] = 0.5

# Color palettes - Professional color scheme
baseline_colormap = plt.cm.Blues     # Blue tones for baselines
model_colormap = plt.cm.tab20c      # Categorical colors for models

# Line and marker styles
baseline_linestyle = "-"   # Solid lines for baselines
model_linestyle = "--"     # Dashed lines for models
marker_styles = ["o", "s", "D", "^", "v", "<", ">", "p", "*", "h", "H", "+"]  # Distinct markers

# Bar patterns for plots without markers
bar_patterns = ['', '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*']

# Create consistent color mapping function with marker assignments
def create_model_color_mapping(all_models):
    """
    Create a consistent color mapping for all models across datasets
    Also assigns markers to models to help distinguish similar colors
    """
    color_mapping = {}
    marker_mapping = {}
    pattern_mapping = {}

    # Separate baseline and non-baseline models
    baseline_models = [m for m in all_models if 'Baseline' in m]
    other_models = [m for m in all_models if 'Baseline' not in m]

    # Sort models for consistency
    baseline_models.sort()
    other_models.sort()

    # Assign baseline colors and markers
    baseline_colors = plt.cm.Blues(np.linspace(0.5, 0.8, len(baseline_models)))
    for i, model in enumerate(baseline_models):
        color_mapping[model] = baseline_colors[i]
        marker_mapping[model] = marker_styles[i % len(marker_styles)]
        pattern_mapping[model] = bar_patterns[i % len(bar_patterns)]

    # Assign other model colors
    # Use tab20c for better contrast in NeurIPS papers
    other_colors = plt.cm.tab20b(np.linspace(0, 1, len(other_models)))

    # Create a color distance matrix to ensure adjacent colors get different markers
    def color_distance(c1, c2):
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1[:3], c2[:3])))

    # Sort models by color similarity
    color_distances = []
    for i, model in enumerate(other_models):
        min_dist = float('inf')
        for j, other_model in enumerate(other_models):
            if i != j:
                dist = color_distance(other_colors[i], other_colors[j])
                if dist < min_dist:
                    min_dist = dist
        color_distances.append((model, i, min_dist))

    # Assign markers ensuring similar colors get different markers
    used_marker_color_pairs = set()
    marker_idx = 0
    pattern_idx = 0

    for model, color_idx, _ in sorted(color_distances, key=lambda x: x[2]):
        color = other_colors[color_idx]
        color_mapping[model] = color

        # Find an appropriate marker that hasn't been used with similar colors
        attempts = 0
        while attempts < len(marker_styles):
            marker = marker_styles[marker_idx % len(marker_styles)]
            pattern = bar_patterns[pattern_idx % len(bar_patterns)]

            # Check if this marker is used with a similar color
            similar_color_uses_marker = False
            for existing_model, existing_color in color_mapping.items():
                if existing_model != model and marker_mapping.get(existing_model) == marker:
                    if color_distance(color, existing_color) < 0.2:  # Threshold for color similarity
                        similar_color_uses_marker = True
                        break

            if not similar_color_uses_marker:
                marker_mapping[model] = marker
                pattern_mapping[model] = pattern
                break

            marker_idx += 1
            pattern_idx += 1
            attempts += 1

        # Fallback if no ideal marker found
        if attempts >= len(marker_styles):
            marker_mapping[model] = marker_styles[marker_idx % len(marker_styles)]
            pattern_mapping[model] = bar_patterns[pattern_idx % len(bar_patterns)]

        marker_idx += 1
        pattern_idx += 1

    return color_mapping, marker_mapping, pattern_mapping

def load_sheets_data(csv_paths):
    """
    Load data from specified Google Sheets with cleaning

    Parameters:
    spreadsheet_name (str): Name of the Google Sheets file

    Returns:
    dict: Dictionary containing cleaned DataFrames for each sheet
    """
    try:
        # Dictionary to store dataframes
        dfs = {}

        # Load data from CSV files
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path)
            dfs[csv_path.stem.split('_')[-1]] = df
        
        datasets = list(dfs.keys())

        # Convert numeric columns to proper type
        for dataset in datasets:
            numeric_cols = dfs[dataset].columns[1:]  # All columns except the first (Model)
            for col in numeric_cols:
                dfs[dataset][col] = pd.to_numeric(dfs[dataset][col], errors='coerce')

        # Clean data: remove rows without model name or avg_score
        for dataset in datasets:
            dfs[dataset] = dfs[dataset][
            (dfs[dataset]['Model'].notna()) &
            (dfs[dataset]['Model'] != '') &
            (dfs[dataset]['avg_score'].notna())
        ].reset_index(drop=True)

        for dataset in datasets:
            print(f"{dataset} loaded: {dfs[dataset].shape[0]} rows (after cleaning), {dfs[dataset].shape[1]} columns")

        # Replace all NaN values with -100.0
        for dataset in datasets:
            dfs[dataset] = dfs[dataset].fillna(-100.0)
            print(f"\nReplaced NaN values with -100.0 in {dataset}")

        return dfs

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def explore_cleaned_data(dfs):
    """
    Explore the cleaned data
    """
    for sheet_name, df in dfs.items():
        print(f"\n{'='*50}")
        print(f"Sheet: {sheet_name}")
        print(f"{'='*50}")

        print(f"\nShape: {df.shape}")
        print(f"\nModels included: {list(df['Model'].values)}")

        print(f"\nFirst 5 rows:")
        display(df.head())

        print(f"\nBasic statistics (numeric columns only):")
        display(df.describe())

        # Check for missing values in task scores
        task_cols = [col for col in df.columns if col.startswith('E')]
        print(f"\nMissing values in task scores (E1-E12):")
        missing_counts = df[task_cols].isnull().sum()
        print(missing_counts[missing_counts > 0])

        print(f"\nAverage score range: {df['avg_score'].min():.2f} to {df['avg_score'].max():.2f}")

def create_model_comparison_plots(data, size=(10, 6), globals=None):
    """
    Create model comparison plots for both KS and Lorenz data

    Parameters:
    data: Dictionary containing dataframes
    size: Tuple of (width, height) in inches
    """
    GLOBAL_MODEL_COLORS, GLOBAL_MODEL_MARKERS, GLOBAL_MODEL_PATTERNS, NEURIPS_DPI = globals

    if not data:
        print("No data loaded")
        return

    # Create side-by-side comparison
    fig, axes = plt.subplots(1, len(data.keys()), figsize=size)

    # Add space at the top for legend
    fig.subplots_adjust(top=0.85)

    # Set fixed x-axis range
    xlim = (-100, 100)

    # Handle case where axes might be a single axis or array of axes
    if len(data.keys()) == 1:
        axes = [axes]
    
    for i, (dataset, ax) in enumerate(zip(data.keys(), axes)):
        dataset_data = data[dataset].sort_values('avg_score', ascending=True)
        bars_pde = axes[i].barh(range(len(dataset_data)), dataset_data['avg_score'])

        # Apply consistent coloring and patterns
        for j, bar in enumerate(bars_pde):
            model_name = dataset_data.iloc[j]['Model']
            bar.set_color(GLOBAL_MODEL_COLORS[model_name])
            bar.set_hatch(GLOBAL_MODEL_PATTERNS[model_name])
            if 'Baseline' in model_name:
                bar.set_edgecolor('black')
                bar.set_linewidth(1.0)

        axes[i].set_yticks(range(len(dataset_data)))
        axes[i].set_yticklabels(dataset_data['Model'])
        axes[i].set_xlabel('Average Score')
        axes[i].set_title(f'{dataset}: Model Performance')
        axes[i].set_xlim(xlim)
        axes[i].grid(axis='x', alpha=0.3)

        # Add value labels on bars
        for j, v in enumerate(dataset_data['avg_score']):
            text_x = v + 2 if v >= 0 else v - 2
            ha = 'left' if v >= 0 else 'right'
            axes[i].text(text_x, j, f'{v:.2f}', va='center', ha=ha)
    # Create a figure-level legend below the title
    # Get all unique models
    all_models = set()
    for dataset in data.keys():
        all_models.update(data[dataset]['Model'])
    all_models = list(all_models)
    all_models.sort()

    # Create legend elements with patterns
    legend_elements = []
    legend_labels = []
    print("All models:", all_models)
    for model in all_models:
        if 'Baseline' in model:
            rect = plt.Rectangle((0,0), 1, 1, facecolor=GLOBAL_MODEL_COLORS[model],
                               edgecolor='black', linewidth=1.0,
                               hatch=GLOBAL_MODEL_PATTERNS[model])
        else:
            rect = plt.Rectangle((0,0), 1, 1, facecolor=GLOBAL_MODEL_COLORS[model],
                               hatch=GLOBAL_MODEL_PATTERNS[model])
        legend_elements.append(rect)
        legend_labels.append(model)

    # Add figure title
    fig.suptitle('Model Average Scores Comparison', y=.95)

    # Add legend below title
    # ncol = 5  # Number of columns in legend
    # legend = fig.legend(legend_elements, legend_labels, loc='upper center',
    #                    bbox_to_anchor=(0.5, 1.05), ncol=ncol, frameon=False)

    plt.tight_layout()
    plt.savefig('model_comparison.pdf', dpi=NEURIPS_DPI, bbox_inches='tight')
    plt.show()

def create_spider_plots(data, n_top_models=4, size=(10, 6), globals=None):
    """
    Create spider (radar) plots showing top N models performance across all tasks
    with properly centered legend values in all rows

    Parameters:
    data: Dictionary containing dataframes
    n_top_models: Number of top models to show
    size: Tuple of (width, height) in inches
    """
    if globals is None:
        print("No globals provided")
        return
    GLOBAL_MODEL_COLORS, GLOBAL_MODEL_MARKERS, GLOBAL_MODEL_PATTERNS, NEURIPS_DPI = globals

    if not data:
        print("No data loaded")
        return

    # Task categories
    categories = [f"E{i+1}" for i in range(12)]
    N = len(categories)

    # Create figure with two subplots
    fig, (axes) = plt.subplots(1, len(data.keys()), figsize=size, subplot_kw=dict(projection="polar"))

    if len(data.keys()) == 1:
        axes = [axes]

    # Adjust layout to make room for legend on the right
    fig.subplots_adjust(top=0.85, right=0.75, wspace=.3)

    # Process each dataset
    all_models_plotted = []

    for ax, sheet_name in zip(axes, data.keys()):
        df = data[sheet_name]

        # Get top N models by average score
        top_models = df.nlargest(n_top_models, 'avg_score')

        # Get baseline (first model in the dataset)
        baseline = df.iloc[0:1]

        # Combine baseline with top models
        all_models = pd.concat([baseline, top_models])

        # Calculate angles
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]

        # Plot each model
        for idx, (_, model) in enumerate(all_models.iterrows()):
            # Get values for E1-E12
            values = [model[f'E{i+1}'] for i in range(12)]
            values += values[:1]  # Close the plot

            # Use global color and marker mapping
            model_name = model['Model']
            color = GLOBAL_MODEL_COLORS[model_name]
            marker = GLOBAL_MODEL_MARKERS[model_name]
            all_models_plotted.append(model_name)

            # Different style for baselines
            if 'Baseline' in model_name:
                linestyle = '--'
                linewidth = 1.5
                marker_size = 5
            else:
                linestyle = '-'
                linewidth = 1.0
                marker_size = 4

            # Plot line and fill
            ax.plot(angles, values, color=color, linestyle=linestyle,
                   linewidth=linewidth, marker=marker, markersize=marker_size)
            ax.fill(angles, values, color=color, alpha=0.1)

        # Customize the plot
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_title(sheet_name, pad=55)

        # Set y-axis limits based on data range
        all_values = all_models[categories].values.flatten()
        y_min = np.nanmin(all_values) - 10
        y_max = np.nanmax(all_values) + 10
        ax.set_ylim(y_min, y_max)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.9)

    # Add figure title
    fig.suptitle(f'Top {n_top_models} Models Characterized on All Metrics', y=1.15)

    # Create horizontal legend below title with proper centering
    unique_models = list(set(all_models_plotted))
    unique_models.sort()

    # Create legend elements
    legend_elements = []
    legend_labels = []
    for model in unique_models:
        marker = GLOBAL_MODEL_MARKERS[model]
        if 'Baseline' in model:
            line = plt.Line2D([0], [0], color=GLOBAL_MODEL_COLORS[model], linewidth=1.5,
                            linestyle='--', marker=marker, markersize=5)
        else:
            line = plt.Line2D([0], [0], color=GLOBAL_MODEL_COLORS[model], linewidth=1.0,
                            linestyle='-', marker=marker, markersize=4)
        legend_elements.append(line)
        legend_labels.append(model)

    # Create legend on the right side as a vertical list
    legend = fig.legend(legend_elements, legend_labels, loc='center right',
                       bbox_to_anchor=(1.0, 0.5), ncol=1, frameon=True,
                       columnspacing=1.0, handletextpad=0.5, handlelength=1.5)

    plt.savefig('spider_plots.pdf', dpi=NEURIPS_DPI, bbox_inches='tight')
    plt.show()

def create_compound_bar_plots(data, n_top_models=3, size=(10, 7), globals=None):
    """
    Create compound bar plots showing top models + baseline for each task
    with patterns for visual distinction

    Parameters:
    data: Dictionary containing dataframes
    n_top_models: Number of top models to show per task
    size: Tuple of (width, height) in inches
    """
    if globals is None:
        print("No globals provided")
        return
    GLOBAL_MODEL_COLORS, GLOBAL_MODEL_MARKERS, GLOBAL_MODEL_PATTERNS, NEURIPS_DPI = globals

    if not data:
        print("No data loaded")
        return

    task_cols = [f'E{i+1}' for i in range(12)]

    # Create figure for each dataset
    for sheet_name in data.keys():
        df = data[sheet_name]

        # Get baseline (first model)
        baseline = df.iloc[0]
        baseline_name = baseline['Model']

        # Create subplots
        fig, axes = plt.subplots(3, 4, figsize=size)

        # Adjust layout for legend at bottom
        fig.subplots_adjust(bottom=0.15, top=0.92)

        axes = axes.flatten()

        for idx, task in enumerate(task_cols):
            ax = axes[idx]

            # Get top 3 models for this task
            task_data = df[['Model', task]].dropna()
            top_models = task_data.nlargest(n_top_models, task)

            # Check if baseline is already in top 3
            if baseline_name in top_models['Model'].values:
                top_models = task_data.nlargest(n_top_models+1, task)
                models = list(top_models['Model'])
                scores = list(top_models[task])
            else:
                models = [baseline_name] + list(top_models['Model'])
                scores = [baseline[task]] + list(top_models[task])

            # Use global color and pattern mapping
            colors = [GLOBAL_MODEL_COLORS[model] for model in models]
            patterns = [GLOBAL_MODEL_PATTERNS[model] for model in models]

            # Create bar plot with patterns
            bars = ax.bar(range(len(models)), scores, color=colors, alpha=0.8,
                         edgecolor='black', linewidth=0.5)

            # Apply patterns
            for bar, pattern in zip(bars, patterns):
                bar.set_hatch(pattern)

            # Highlight baselines
            for i, model in enumerate(models):
                if 'Baseline' in model:
                    bars[i].set_edgecolor('black')
                    bars[i].set_linewidth(1.0)

            # Customize the plot
            # ax.set_xticks(range(len(models)))
            # ax.set_xticklabels(models, rotation=45, ha='right')
            ax.set_xticklabels([])
            ax.set_ylabel('Score')
            ax.set_title(f'{task}')
            ax.grid(axis='y', alpha=0.3)

            # Add value labels on bars
            for bar, score in zip(bars, scores):
                if pd.notna(score):
                    height = bar.get_height()
                    if height >= 0:
                        va = 'bottom'
                        y_offset = 1
                    else:
                        va = 'top'
                        y_offset = -1
                    ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                           f'{score:.1f}', ha='center', va=va)

        # Add figure title
        fig.suptitle(f'{sheet_name}: Top {n_top_models} Models + Baseline per Metric', y=0.98)

        # Create legend at bottom with patterns
        appearing_models = set()
        for task in task_cols:
            task_data = df[['Model', task]].dropna()
            top_models = task_data.nlargest(n_top_models, task)
            appearing_models.update(top_models['Model'].values)
        appearing_models.add(baseline_name)

        # Create legend elements with patterns
        legend_elements = []
        legend_labels = []
        for model in sorted(appearing_models):
            edge_width = 1.0 if 'Baseline' in model else 0.5
            rect = plt.Rectangle((0,0), 1, 1, facecolor=GLOBAL_MODEL_COLORS[model],
                               edgecolor='black', linewidth=edge_width, alpha=0.8,
                               hatch=GLOBAL_MODEL_PATTERNS[model])
            legend_elements.append(rect)
            legend_labels.append(model)

        # Place legend at bottom
        ncol = min(5, len(appearing_models))
        fig.legend(legend_elements, legend_labels,
                  title='Models', loc='upper center',
                  bbox_to_anchor=(0.5, 0.02), ncol=ncol, frameon=True)

        plt.tight_layout()
        plt.savefig(f'{sheet_name}_compound_bars.pdf', dpi=NEURIPS_DPI, bbox_inches='tight')
        plt.show()

def create_compact_compound_bar_plots(data, n_top_models=3, size=(10, 5), globals=None):
    """
    Create compact compound bar plots showing top models + baseline for each task
    Uses 2 rows with 6 columns each and shared y-axes
    with patterns for visual distinction

    Parameters:
    data: Dictionary containing dataframes
    n_top_models: Number of top models to show per task
    size: Tuple of (width, height) in inches
    """
    if globals is None:
        print("No globals provided")
        return
    GLOBAL_MODEL_COLORS, GLOBAL_MODEL_MARKERS, GLOBAL_MODEL_PATTERNS, NEURIPS_DPI = globals

    if not data:
        print("No data loaded")
        return

    task_cols = [f'E{i+1}' for i in range(12)]

    # Create figure for each dataset
    # Track figure index for enumeration
    for fig_idx, sheet_name in enumerate(data.keys()):
        df = data[sheet_name]

        # Get baseline (first model)
        baseline = df.iloc[0]
        baseline_name = baseline['Model']

        # Create subplots with shared y-axes
        fig, axes = plt.subplots(2, 6, figsize=size, sharey=True)

        # Adjust layout for legend at bottom
        fig.subplots_adjust(bottom=0.2, top=0.92, wspace=0.05, hspace=0.4)

        axes = axes.flatten()

        for idx, task in enumerate(task_cols):
            ax = axes[idx]

            # Get top 3 models for this task
            task_data = df[['Model', task]].dropna()
            top_models = task_data.nlargest(n_top_models, task)

            # Check if baseline is already in top 3
            if baseline_name in top_models['Model'].values:
                top_models = task_data.nlargest(n_top_models+1, task)
                models = list(top_models['Model'])
                scores = list(top_models[task])
            else:
                models = [baseline_name] + list(top_models['Model'])
                scores = [baseline[task]] + list(top_models[task])

            # Use global color and pattern mapping
            colors = [GLOBAL_MODEL_COLORS[model] for model in models]
            patterns = [None for model in models] #[GLOBAL_MODEL_PATTERNS[model] for model in models]

            # Create bar plot with patterns
            bars = ax.bar(range(len(models)), scores, color=colors, alpha=0.8,
                         edgecolor='black', linewidth=0.5)

            # Apply patterns
            for bar, pattern in zip(bars, patterns):
                bar.set_hatch(pattern)

            # Highlight baselines
            for i, model in enumerate(models):
                if 'Baseline' in model:
                    bars[i].set_edgecolor('black')
                    bars[i].set_linewidth(1.0)

            # Customize the plot
            ax.set_ylim(-109, 109)  # Fixed y-axis range
            ax.set_xticks([])  # No x-axis labels
            ax.set_title(f'{task}')
            ax.grid(axis='y', alpha=0.3)

            # Only show y-axis labels on leftmost plots
            if idx % 6 == 0:
                ax.set_ylabel('Score')

            # Add value labels on bars
            for bar, score in zip(bars, scores):
                if pd.notna(score):
                    height = bar.get_height()
                    if height >= 0:
                        va = 'bottom'
                        y_offset = 1
                    else:
                        va = 'top'
                        y_offset = -1
                    ax.text(bar.get_x() + bar.get_width()/2., height + y_offset,
                           f'{score:.0f}', ha='center', va=va, fontsize=7)

        # Add figure title
        #fig.suptitle(f'{sheet_name}: Top {n_top_models} Models per Metric', y=0.98)

        # Create legend at bottom with patterns
        appearing_models = set()
        for task in task_cols:
            task_data = df[['Model', task]].dropna()
            extended_top = task_data.nlargest(n_top_models + 1, task)

            if baseline_name in extended_top['Model'].values[:n_top_models]:
                appearing_models.update(extended_top['Model'].values[:n_top_models])
            else:
                top_models = task_data.nlargest(n_top_models, task)
                appearing_models.update(top_models['Model'].values)
                appearing_models.add(baseline_name)

        # Create legend elements with patterns
        legend_elements = []
        legend_labels = []
        for model in sorted(appearing_models):
            edge_width = 1.0 if 'Baseline' in model else 0.5
            rect = plt.Rectangle((0,0), 1, 1, facecolor=GLOBAL_MODEL_COLORS[model],
                               edgecolor='black', linewidth=edge_width, alpha=0.8,
                               #hatch=GLOBAL_MODEL_PATTERNS[model]
                                 )
            legend_elements.append(rect)
            legend_labels.append(model)

        # Place legend at bottom
        ncol = min(5, len(appearing_models))
        fig.legend(legend_elements, legend_labels,
                  title='Models', loc='upper center',
                  bbox_to_anchor=(0.5, 0.05), ncol=ncol, frameon=True)

        # Add letter enumeration in the top left corner
        # Get the letter based on figure index: 0 -> 'a', 1 -> 'b'
        letter = chr(97 + fig_idx)  # 97 is ASCII for 'a'
        fig.text(0.02, 0.98, f'({letter})', fontsize=10, fontweight='bold',
                 horizontalalignment='left', verticalalignment='top')

        plt.tight_layout()
        plt.savefig(f'{sheet_name}_compact_bars.pdf', dpi=NEURIPS_DPI, bbox_inches='tight')
        plt.show()

def create_ultra_compact_bar_plots(data, n_top_models=3, size=(10, 5), use_patterns=True, globals=None):
    """
    Create ultra-compact compound bar plots showing top models with baseline as a dashed line
    Uses 2 rows with 6 columns each and shared y-axes
    Model names are written directly on the plot, eliminating the need for a legend
    Adds letter enumeration (a) for PDE_KS and (b) for ODE_Lorenz in the top left corner

    Parameters:
    data: Dictionary containing dataframes
    n_top_models: Number of top models to show per task
    size: Tuple of (width, height) in inches
    use_patterns: Boolean flag to enable/disable patterns on bars
    """
    if globals is None:
        print("No globals provided")
        return
    GLOBAL_MODEL_COLORS, GLOBAL_MODEL_MARKERS, GLOBAL_MODEL_PATTERNS, NEURIPS_DPI = globals

    if not data:
        print("No data loaded")
        return

    task_cols = [f'E{i+1}' for i in range(12)]

    # Create figure for each dataset
    # Track figure index for enumeration
    for fig_idx, sheet_name in enumerate(data.keys()):
        df = data[sheet_name]

        # Get baseline (first model)
        baseline = df.iloc[0]
        baseline_name = baseline['Model']

        # Create subplots with shared y-axes
        fig, axes = plt.subplots(2, 6, figsize=size, sharey=True)

        # Adjust layout - no need for legend space
        fig.subplots_adjust(bottom=0.05, top=0.92, wspace=0.05, hspace=0.4)

        axes = axes.flatten()

        for idx, task in enumerate(task_cols):
            ax = axes[idx]

            # Get top 3 models for this task (excluding baseline)
            task_data = df[['Model', task]].dropna()
            # Exclude baseline from the top models
            non_baseline_data = task_data[task_data['Model'] != baseline_name]
            top_models = non_baseline_data.nlargest(n_top_models, task)

            models = list(top_models['Model'])
            scores = list(top_models[task])

            # Use global color and pattern mapping
            colors = [GLOBAL_MODEL_COLORS[model] for model in models]
            patterns = [GLOBAL_MODEL_PATTERNS[model] for model in models] if use_patterns else ['' for _ in models]

            # Create bar plot with patterns
            bars = ax.bar(range(len(models)), scores, color=colors, alpha=0.8,
                         edgecolor='black', linewidth=0.5)

            # Apply patterns if enabled
            if use_patterns:
                for bar, pattern in zip(bars, patterns):
                    bar.set_hatch(pattern)

            # Add baseline as dashed horizontal line
            baseline_color = plt.colormaps['Blues'](0.87)
            baseline_score = baseline[task]
            ax.axhline(y=baseline_score, color=baseline_color, linestyle='--', linewidth=1.25, alpha=0.5)

            # Add baseline label at the right side of the plot
            ax.text(len(models) - 0.1, baseline_score + 2, "baseline",
                   rotation=90, ha='right', va='center', fontsize=6, color=baseline_color)

            # Customize the plot
            ax.set_ylim(-109, 109)  # Fixed y-axis range
            ax.set_xticks([])  # No x-axis labels
            ax.set_title(f'{task}')
            ax.grid(axis='y', alpha=0.3)

            # Only show y-axis labels on leftmost plots
            if idx % 6 == 0:
                ax.set_ylabel('Score')

            # Add model names directly on the plot
            for i, (bar, model, score) in enumerate(zip(bars, models, scores)):
                if pd.notna(score):
                    height = bar.get_height()

                    # Determine text position based on bar height
                    if height >= 0:
                        # Positive bars: text below
                        va = 'top'
                        y_pos = -2  # Just below x-axis
                    else:
                        # Negative bars: text above
                        va = 'bottom'
                        y_pos = height + 2

                    # Add model name with 90-degree rotation
                    ax.text(bar.get_x() + bar.get_width()/2., y_pos, model,
                           rotation=90, ha='center', va=va, fontsize=6)

                    # Add score value on the bar
                    if height >= 0:
                        score_va = 'bottom'
                        score_y = height + 1
                    else:
                        score_va = 'top'
                        score_y = height - 1

                    ax.text(bar.get_x() + bar.get_width()/2., score_y,
                           f'{score:.0f}', ha='center', va=score_va, fontsize=7)

        # Add figure title
        #fig.suptitle(f'{sheet_name}: Top {n_top_models} Models per Metric (Baseline: {baseline_name})', y=0.98)

        # Add letter enumeration in the top left corner
        # Get the letter based on figure index: 0 -> 'a', 1 -> 'b'
        letter = chr(97 + fig_idx)  # 97 is ASCII for 'a'
        fig.text(0.02, 0.97, f'({letter})', fontsize=9, fontweight='bold',
                 horizontalalignment='left', verticalalignment='top')

        plt.tight_layout()
        plt.savefig(f'{sheet_name}_ultra_compact_bars.pdf', dpi=NEURIPS_DPI, bbox_inches='tight')
        plt.show()

def create_task_performance_heatmaps(data, size=(10, 6), globals=None):
    """
    Create heatmaps showing model performance across all tasks

    Parameters:
    data: Dictionary containing dataframes
    size: Tuple of (width, height) in inches
    """
    if globals is None:
        print("No globals provided")
        return
    GLOBAL_MODEL_COLORS, GLOBAL_MODEL_MARKERS, GLOBAL_MODEL_PATTERNS, NEURIPS_DPI = globals

    if not data:
        print("No data loaded")
        return

    fig, axes = plt.subplots(1, len(data.keys()), figsize=size)

    if len(data.keys()) == 1:
        axes = [axes]

    # Add space at the top for title
    fig.subplots_adjust(top=0.92)

    for ax, sheet_name in zip(axes, data.keys()):
        # heatmap
        task_cols = [col for col in data[sheet_name].columns if col.startswith('E')]
        pde_task_data = data[sheet_name][['Model'] + task_cols].set_index('Model')

        im1 = ax.imshow(pde_task_data.values, cmap='RdYlGn', aspect='auto')
        ax.set_xticks(range(len(task_cols)))
        ax.set_xticklabels(task_cols, rotation=45)
        ax.set_yticks(range(len(pde_task_data)))
        ax.set_yticklabels(pde_task_data.index)
        ax.set_title('PDE_KS: Task Performance Heatmap')

        # Add colorbar with smaller size
        cbar1 = plt.colorbar(im1, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label('Score')

    # Add figure title
    fig.suptitle('Model Performance Across Tasks (E1-E12)', y=0.98)

    plt.tight_layout()
    plt.savefig('task_heatmaps.pdf', dpi=NEURIPS_DPI, bbox_inches='tight')
    plt.show()

def create_interactive_task_comparison(data):
    """
    Create interactive plots comparing model performance on each task
    """

    if not data:
        print("No data loaded")
        return

    task_cols = [col for col in data[list(data.keys())[0]].columns if col.startswith('E')]

    # Create subplots for each task
    fig = make_subplots(
        rows=3, cols=4,
        subplot_titles=[f'Task {task}' for task in task_cols],
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )

    for idx, task in enumerate(task_cols):
        row = idx // 4 + 1
        col = idx % 4 + 1

        for sheet_name in data.keys():
            scores = data[sheet_name][['Model', task]].dropna()
            fig.add_trace(
                go.Bar(
                    x=scores['Model'],
                    y=scores[task],
                    name=f'{str(sheet_name)}',
                    marker_color='blue',
                    opacity=0.7,
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )

    fig.update_layout(
        height=1000,
        title_text="Model Performance Comparison Across All Tasks",
        showlegend=True,
        barmode='group'
    )

    fig.update_xaxes(tickangle=45)
    fig.show()

def statistical_analysis(data, size=(10, 8), globals=None):
    """
    Perform statistical analysis of model performance

    Parameters:
    data: Dictionary containing dataframes
    size: Tuple of (width, height) in inches
    """
    if globals is None:
        print("No globals provided")
        return
    GLOBAL_MODEL_COLORS, GLOBAL_MODEL_MARKERS, GLOBAL_MODEL_PATTERNS, NEURIPS_DPI = globals

    if not data:
        print("No data loaded")
        return

    # Create comparison plots for both datasets
    fig, axes = plt.subplots(2, len(data.keys()), figsize=size)

    if len(data.keys()) == 1:
        axes = axes.reshape(1, 2)

    # Adjust for title
    fig.subplots_adjust(top=0.92)

    task_cols = [col for col in data[list(data.keys())[0]].columns if col.startswith('E')]

    for sheet_num, sheet_name in enumerate(data.keys()):
        task_data = data[sheet_name][task_cols]

        # Box plot of task score distributions
        print(type(axes))
        task_data.plot(kind='box', ax=axes[sheet_num, 0], rot=45)
        axes[sheet_num, 0].set_title(f'{sheet_name}: Task Score Distributions')
        axes[sheet_num, 0].set_ylabel('Score')

        # Violin plot of model scores across tasks

        print("1")
        model_scores_ode = data[sheet_name][task_cols].values.flatten()
        model_names_ode = np.repeat(data[sheet_name]['Model'].values, len(task_cols))
        task_names_ode = np.tile(task_cols, len(data[sheet_name]))

        ode_long_df = pd.DataFrame({
            'Model': model_names_ode,
            'Task': task_names_ode,
            'Score': model_scores_ode
        }).dropna()

        sns.violinplot(data=ode_long_df, x='Task', y='Score', ax=axes[sheet_num, 1], linewidth=0.5)
        axes[sheet_num, 1].set_title(f'{sheet_name}: Score Distribution by Task')
        axes[sheet_num, 1].tick_params(axis='x', rotation=45)

    # Add figure title
    fig.suptitle('Statistical Analysis of Model Performance', y=0.98)

    plt.tight_layout()
    plt.savefig('statistical_analysis.pdf', dpi=NEURIPS_DPI, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\nSummary Statistics:")
    print("="*50)

    for sheet_name, df in data.items():
        print(f"\n{sheet_name}:")
        print(f"  Number of models: {len(df)}")
        print(f"  Average score range: {df['avg_score'].min():.2f} to {df['avg_score'].max():.2f}")
        print(f"  Mean of average scores: {df['avg_score'].mean():.2f}")
        print(f"  Std of average scores: {df['avg_score'].std():.2f}")

        # Best and worst performing models
        best_model = df.loc[df['avg_score'].idxmax()]
        worst_model = df.loc[df['avg_score'].idxmin()]

        print(f"  Best performing model: {best_model['Model']} (score: {best_model['avg_score']:.2f})")
        print(f"  Worst performing model: {worst_model['Model']} (score: {worst_model['avg_score']:.2f})")

def create_model_rankings(data, globals=None):
    """
    Create visualizations showing model rankings
    """
    if globals is None:
        print("No globals provided")
        return
    GLOBAL_MODEL_COLORS, GLOBAL_MODEL_MARKERS, GLOBAL_MODEL_PATTERNS, NEURIPS_DPI = globals

    if not data:
        print("No data loaded")
        return

    # Create interactive ranking plot
    fig = make_subplots(
        rows=1, cols=len(data.keys()),
        subplot_titles=[f'{sheet_name} Model Rankings' for sheet_name in data.keys()],
        specs=[[{"type": "scatter"}] * len(data.keys())]
    )

    for sheet_num, sheet_name in enumerate(data.keys()):

        data_ranked = data[sheet_name].sort_values('avg_score', ascending=False).reset_index(drop=True)
        data_ranked['rank'] = range(1, len(data_ranked) + 1)

        fig.add_trace(
            go.Scatter(
                x=data_ranked['rank'],
                y=data_ranked['avg_score'],
                mode='markers+text',
                text=data_ranked['Model'],
                textposition="top center",
                marker=dict(
                    size=12,
                    color=data_ranked['avg_score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Avg Score", x=1.05)
                ),
                name=f'{sheet_name} Models'
            ),
            row=1, col=sheet_num + 1
        )

        fig.update_xaxes(title_text="Rank", row=1, col=sheet_num + 1)
        fig.update_yaxes(title_text="Average Score", row=1, col=sheet_num + 1)

    fig.update_layout(
        height=600,
        showlegend=False,
        title_text="Model Rankings by Average Score"
    )

    fig.show()

    # Print ranking tables
    print("\nModel Rankings:")
    print("="*50)

    for sheet_name, df in data.items():

        data_ranked = df.sort_values('avg_score', ascending=False).reset_index(drop=True)
        data_ranked['rank'] = range(1, len(data_ranked) + 1)

        print(f"\n{sheet_name} Rankings:")
        print(data_ranked[['rank', 'Model', 'avg_score']].to_string(index=False))

def create_model_rankings_matplotlib(data, size=(6.5, 3.5), globals=None):
    """
    Create model ranking visualization using Matplotlib with angled labels

    Parameters:
    data: Dictionary containing dataframes
    size: Tuple of (width, height) in inches
    """
    if globals is None:
        print("No globals provided")
        return
    GLOBAL_MODEL_COLORS, GLOBAL_MODEL_MARKERS, GLOBAL_MODEL_PATTERNS, NEURIPS_DPI = globals

    if not data:
        print("No data loaded")
        return

    # Create figure
    fig, axes = plt.subplots(1, len(data.keys()), figsize=size)

    if len(data.keys()) == 1:
        axes = [axes]

    # Adjust layout
    fig.subplots_adjust(top=0.9, bottom=0.1)

    # Scatter point size
    scatter_size = 75

    for sheet_num, sheet_name in enumerate(data.keys()):
        plot_ranking_scatter(axes[sheet_num], data[sheet_name], f'{sheet_name} Model Rankings',
                        'viridis', scatter_size, fig, globals)

    # Add figure title
    fig.suptitle('Model Rankings by Average Score', y=0.98)

    plt.tight_layout()
    plt.savefig('model_rankings_matplotlib.pdf', dpi=NEURIPS_DPI, bbox_inches='tight')
    plt.show()

    # Print ranking tables
    print_ranking_tables(data)

def plot_ranking_scatter(ax, df, title, colormap, scatter_size, fig, globals=None):
    """
    Create a single ranking scatter plot with markers

    Parameters:
    ax: Matplotlib axis object
    df: DataFrame with model data
    title: Plot title
    colormap: Colormap name for scatter points
    scatter_size: Size of scatter points
    fig: Figure object for offset calculation
    """
    if globals is None:
        print("No globals provided")
        return
    GLOBAL_MODEL_COLORS, GLOBAL_MODEL_MARKERS, GLOBAL_MODEL_PATTERNS, NEURIPS_DPI = globals

    # Rank models by average score
    ranked = df.sort_values('avg_score', ascending=False).reset_index(drop=True)
    ranked['rank'] = range(1, len(ranked) + 1)

    # Create scatter plot with markers
    for idx, row in ranked.iterrows():
        model_name = row['Model']
        marker = GLOBAL_MODEL_MARKERS[model_name]
        ax.scatter(row['rank'], row['avg_score'],
                  c=[ranked['avg_score'].iloc[idx]], cmap=colormap,
                  s=scatter_size, alpha=0.8, edgecolors='black', linewidth=0.5,
                  marker=marker, vmin=ranked['avg_score'].min(), vmax=ranked['avg_score'].max())

    # Add colormap
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=ranked['avg_score'].min(),
                                                                vmax=ranked['avg_score'].max()))
    sm.set_array([])

    # Calculate label offset
    y_offset = calculate_label_offset(ax, fig, scatter_size)

    # Add text labels
    add_text_labels(ax, ranked, y_offset)

    # Configure axis
    ax.set_xlabel('Rank')
    ax.set_ylabel('Average Score')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-100, 100)

    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, shrink=0.6, location='right')
    #cbar.set_label('Avg Score', fontsize=8)
    cbar.ax.tick_params(labelsize=7)

def calculate_label_offset(ax, fig, scatter_size):
    """
    Calculate the offset for text labels based on scatter point size

    Parameters:
    ax: Matplotlib axis object
    fig: Figure object
    scatter_size: Size of scatter points

    Returns:
    float: Offset in data coordinates
    """
    # Convert scatter size to approximate radius in data coordinates
    point_radius_inches = np.sqrt(scatter_size) / 72.0  # 72 points per inch
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    y_offset = 1.75 * point_radius_inches * y_range / (ax.get_position().height * fig.get_figheight())
    return y_offset

def add_text_labels(ax, ranked_df, y_offset):
    """
    Add text labels to scatter points with appropriate positioning

    Parameters:
    ax: Matplotlib axis object
    ranked_df: DataFrame with ranked models
    y_offset: Offset for text positioning
    """
    for idx, row in ranked_df.iterrows():
        if False: #idx > len(ranked_df)/2:
            # Top half - place label above point
            va = 'bottom'
            y_pos = row['avg_score'] + y_offset
        else:
            # Bottom half - place label below point
            va = 'top'
            y_pos = row['avg_score'] - y_offset

        ax.text(row['rank'], y_pos, row['Model'],
                rotation=90, ha='center', va=va, fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.1))

def print_ranking_tables(data):
    """
    Print ranking tables for both datasets

    Parameters:
    data: Dictionary containing dataframes
    """
    print("\nModel Rankings:")
    print("="*50)

    for sheet_name, df in data.items():
        ranked = df.sort_values('avg_score', ascending=False).reset_index(drop=True)
        ranked['rank'] = range(1, len(ranked) + 1)

        print(f"\n{sheet_name} Rankings:")
        print(ranked[['rank', 'Model', 'avg_score']].to_string(index=False))
        print()

def create_latex_performance_tables(data):
    """
    Create LaTeX tables showing model rankings with best scores highlighted
    """
    if not data:
        print("No data loaded")
        return

    import pandas as pd
    from IPython.display import display, Markdown

    latex_tables = {}

    for sheet_name, df in data.items():
        # Sort by average score in descending order
        df_sorted = df.sort_values('avg_score', ascending=False).copy()

        # Find the best score for each metric
        task_cols = [f'E{i+1}' for i in range(12)]
        best_scores = {}
        for col in task_cols:
            best_scores[col] = df_sorted[col].max()

        # Format the dataframe for LaTeX
        latex_df = df_sorted[['Model', 'avg_score'] + task_cols].copy()

        # Round numeric values
        numeric_cols = ['avg_score'] + task_cols
        for col in numeric_cols:
            latex_df[col] = latex_df[col].round(2)

        # Create LaTeX string with highlighting
        def highlight_best(val, col):
            if col in best_scores and np.isclose(val, best_scores[col], rtol=1e-3, atol=1e-3):
                return f'\\textbf{{{val}}}'
            else:
                return str(val)

        # Apply highlighting
        for col in task_cols:
            latex_df[col] = latex_df.apply(lambda row: highlight_best(row[col], col), axis=1)

        # Create the LaTeX table
        latex_str = f"\\begin{{table}}[h!]\n"
        latex_str += f"\\centering\n"
        latex_str += f"\\caption{{Model Performance Rankings: {sheet_name}}}\n"
        latex_str += f"\\label{{tab:{sheet_name.lower()}_rankings}}\n"
        latex_str += f"\\begin{{adjustbox}}{{width=\\textwidth}}\n"
        latex_str += f"\\begin{{tabular}}{{l|r|{'r' * 12}}}\n"
        latex_str += f"\\hline\n"
        latex_str += f"\\textbf{{Model}} & \\textbf{{Avg Score}} & "
        latex_str += " & ".join([f"\\textbf{{{col}}}" for col in task_cols]) + " \\\\\n"
        latex_str += f"\\hline\n"

        # Add data rows
        for _, row in latex_df.iterrows():
            latex_str += f"{row['Model']} & {row['avg_score']} & "
            latex_str += " & ".join([str(row[col]) for col in task_cols]) + " \\\\\n"

        latex_str += f"\\hline\n"
        latex_str += f"\\end{{tabular}}\n"
        latex_str += f"\\end{{adjustbox}}\n"
        latex_str += f"\\end{{table}}\n"

        latex_tables[sheet_name] = latex_str

        # Display the table
        print(f"\n{'=' * 50}")
        print(f"LaTeX Table for {sheet_name}")
        print('=' * 50)
        print(latex_str)

        # Also create a markdown preview
        print(f"\nPreview for {sheet_name}:")
        display_df = df_sorted[['Model', 'avg_score'] + task_cols].copy()

        # Format with asterisks for markdown preview
        for col in task_cols:
            display_df[col] = display_df[col].round(2)
            mask = display_df[col] == best_scores[col]
            display_df.loc[mask, col] = display_df.loc[mask, col].astype(str) + '*'

        display(display_df)
        print("Note: * indicates the best score for that task")

    # Save LaTeX tables to files
    for sheet_name, latex_str in latex_tables.items():
        filename = f"{sheet_name}_rankings_table.tex"
        with open(filename, 'w') as f:
            f.write(latex_str)
        print(f"\nSaved LaTeX table to {filename}")

    # Create a combined LaTeX document
    combined_latex = "\\documentclass{article}\n"
    combined_latex += "\\usepackage{adjustbox}\n"
    combined_latex += "\\usepackage{booktabs}\n"
    combined_latex += "\\begin{document}\n\n"

    for sheet_name, latex_str in latex_tables.items():
        combined_latex += latex_str + "\n\\clearpage\n\n"

    combined_latex += "\\end{document}"

    with open("model_rankings_combined.tex", 'w') as f:
        f.write(combined_latex)
    print("\nSaved combined LaTeX document to model_rankings_combined.tex")

    return latex_tables

def save_analysis_results(data, save_path=''):
    """
    Save processed data and analysis results
    """
    if not data:
        print("No data to save")
        return

    # Save cleaned data
    for sheet_name, df in data.items():
        filename = f"{save_path}{sheet_name}_cleaned.csv"
        df.to_csv(filename, index=False)
        print(f"Saved cleaned {sheet_name} to {filename}")

    # Create summary report
    with open(f"{save_path}model_performance_summary.txt", 'w') as f:
        f.write("Model Performance Summary Report\n")
        f.write("="*50 + "\n\n")

        for sheet_name, df in data.items():
            f.write(f"{sheet_name} Analysis:\n")
            f.write("-"*30 + "\n")
            f.write(f"Number of models: {len(df)}\n")
            f.write(f"Average score range: {df['avg_score'].min():.2f} to {df['avg_score'].max():.2f}\n")
            f.write(f"Mean of average scores: {df['avg_score'].mean():.2f}\n")
            f.write(f"Std of average scores: {df['avg_score'].std():.2f}\n")

            best_model = df.loc[df['avg_score'].idxmax()]
            worst_model = df.loc[df['avg_score'].idxmin()]

            f.write(f"Best performing model: {best_model['Model']} (score: {best_model['avg_score']:.2f})\n")
            f.write(f"Worst performing model: {worst_model['Model']} (score: {worst_model['avg_score']:.2f})\n\n")

    print(f"\nSaved summary report to {save_path}model_performance_summary.txt")


