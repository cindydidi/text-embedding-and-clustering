#!/usr/bin/env python3
"""
Step 5: Create visualizations for cluster analysis.
Uses cluster_comparison.csv from step 4; group and column names are read from the data.
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
import os

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

COMPARISON_CSV = 'cluster_comparison.csv'
COMPARISON_REPORT = 'cluster_comparison_report.txt'
METADATA_CSV = 'embeddings_metadata.csv'
CLUSTERS_CSV = 'clusters_with_labels.csv'


def get_categories_from_comparison(comparison_df):
    """Infer group category names from comparison CSV columns (*_Count)."""
    count_cols = [c for c in comparison_df.columns if c.endswith('_Count')]
    return [c.replace('_Count', '') for c in count_cols]


def get_group_col_from_report(report_path=COMPARISON_REPORT):
    """Parse group-by column name from step 4 report. Returns None if not found."""
    if not os.path.isfile(report_path):
        return None
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            text = f.read()
        m = re.search(r"CLUSTER COMPARISON BY\s+'([^']+)'", text)
        return m.group(1).strip() if m else None
    except Exception:
        return None


def load_data():
    """Load all necessary data files."""
    print("Loading data...")
    comparison_df = pd.read_csv(COMPARISON_CSV)
    metadata = pd.read_csv(METADATA_CSV)
    clusters_df = pd.read_csv(CLUSTERS_CSV)
    print("✓ Data loaded")
    return comparison_df, metadata, clusters_df


def create_cluster_comparison_chart(comparison_df, categories, output_file):
    """Create bar chart comparing clusters across groups (data-driven)."""
    if not categories:
        print("  Skip comparison chart: no group columns found")
        return
    print("Creating cluster comparison chart...")

    count_cols = [f'{c}_Count' for c in categories]
    comparison_df = comparison_df.copy()
    comparison_df['Total_Count'] = comparison_df[count_cols].sum(axis=1)
    top_clusters = comparison_df.nlargest(15, 'Total_Count')

    percent_cols = [f'{c}_Percent' for c in categories]
    for c in percent_cols:
        if c not in top_clusters.columns:
            print(f"  Skip comparison chart: missing column {c}")
            return

    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(top_clusters))
    n_groups = len(categories)
    width = 0.8 / n_groups
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_groups, 2)))

    for i, (cat, pct_col) in enumerate(zip(categories, percent_cols)):
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, top_clusters[pct_col], width, label=cat, color=colors[i], alpha=0.8)

    ax.set_xlabel('Clusters', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage of Messages (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Cluster Distribution by Group ({", ".join(categories)})', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(top_clusters['Cluster_Label'], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_file}")
    plt.close()


def create_wordcloud(metadata, group_col, channel, output_file):
    """Create word cloud for a specific group value."""
    print(f"Creating word cloud for {channel}...")
    col = metadata.get(group_col)
    if col is None:
        print(f"  Warning: Column '{group_col}' not in metadata; skip word cloud for {channel}")
        return
    channel_messages = metadata[metadata[group_col].astype(str).str.strip() == str(channel)]['Message'].dropna().astype(str).tolist()
    if not channel_messages:
        print(f"  Warning: No messages found for {channel}")
        return
    text = ' '.join(channel_messages)
    wordcloud = WordCloud(
        width=1200, height=600, background_color='white',
        max_words=100, colormap='viridis', relative_scaling=0.5, collocations=False
    ).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Word Cloud: {channel}', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_file}")
    plt.close()


def create_cluster_heatmap(comparison_df, categories, output_file):
    """Create heatmap of cluster percentages by group (data-driven)."""
    if not categories:
        print("  Skip heatmap: no group columns found")
        return
    print("Creating cluster heatmap...")
    count_cols = [f'{c}_Count' for c in categories]
    comparison_df = comparison_df.copy()
    comparison_df['Total_Count'] = comparison_df[count_cols].sum(axis=1)
    top_clusters = comparison_df.nlargest(20, 'Total_Count')

    percent_cols = [f'{c}_Percent' for c in categories]
    if not all(c in top_clusters.columns for c in percent_cols):
        print("  Skip heatmap: missing percent columns")
        return
    heatmap_data = top_clusters[['Cluster_Label'] + percent_cols].copy()
    heatmap_data = heatmap_data.set_index('Cluster_Label')
    heatmap_data.columns = categories

    fig, ax = plt.subplots(figsize=(10, 12))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
                cbar_kws={'label': 'Percentage (%)'}, ax=ax, linewidths=0.5, linecolor='gray')
    ax.set_title(f'Cluster Distribution by Group ({", ".join(categories)})', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Clusters', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_file}")
    plt.close()


def create_difference_chart(comparison_df, categories, output_file):
    """Create chart of significant differences (only when 2 groups)."""
    if 'Difference' not in comparison_df.columns or 'P_Value' not in comparison_df.columns:
        print("  Skip difference chart: only available for 2-group comparison")
        return
    print("Creating difference chart...")
    significant = comparison_df[comparison_df['P_Value'] < 0.05].copy()
    significant = significant.sort_values('Difference', key=abs, ascending=False).head(15)
    if len(significant) == 0:
        print("  No significant differences found for chart")
        return

    diff_label = f"{categories[0]} - {categories[1]}" if len(categories) == 2 else "Difference"
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#2E86AB' if d > 0 else '#A23B72' for d in significant['Difference']]
    ax.barh(significant['Cluster_Label'], significant['Difference'], color=colors, alpha=0.8)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel(f'Difference in Percentage ({diff_label})', fontsize=12, fontweight='bold')
    ax.set_ylabel('Clusters', fontsize=12, fontweight='bold')
    ax.set_title('Significant Cluster Differences Between Groups', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved to {output_file}")
    plt.close()


if __name__ == "__main__":
    try:
        comparison_df, metadata, clusters_df = load_data()
        categories = get_categories_from_comparison(comparison_df)
        group_col = get_group_col_from_report()

        print("\n" + "="*60)
        print("CREATING VISUALIZATIONS")
        print("="*60)
        print(f"Groups from comparison: {categories}")
        print(f"Group-by column (for word clouds): {group_col or 'not found (skip word clouds)'}")

        generated = []
        create_cluster_comparison_chart(comparison_df, categories, 'cluster_comparison_chart.png')
        generated.append('cluster_comparison_chart.png')

        if group_col and group_col in metadata.columns:
            for cat in categories:
                safe = re.sub(r'[^\w\-]', '_', str(cat))
                out = f'wordcloud_{safe}.png'
                create_wordcloud(metadata, group_col, cat, out)
                generated.append(out)
        else:
            print("  Skipping word clouds (no group column in report or metadata)")

        create_cluster_heatmap(comparison_df, categories, 'cluster_heatmap.png')
        generated.append('cluster_heatmap.png')
        create_difference_chart(comparison_df, categories, 'cluster_differences_chart.png')
        generated.append('cluster_differences_chart.png')

        print("\n" + "="*60)
        print("Step 5 Complete! All visualizations created.")
        print("="*60)
        print("\nGenerated files:")
        for f in generated:
            print(f"  - {f}")
        print("="*60)

    except FileNotFoundError as e:
        print(f"Error: {e} not found. Please run previous steps first.")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
