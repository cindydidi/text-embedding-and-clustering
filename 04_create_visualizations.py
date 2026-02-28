#!/usr/bin/env python3
"""
Step 4: Create reports and visualizations for cluster analysis.

Default mode (no flags):
- Produces a fixed set of outputs (overall summary + comparison charts).

LLM mode (--llm):
- Skips the fixed outputs and instead prompts the user for an ad-hoc request
  (numbers or visualizations), then uses Gemini to generate and run code.
"""

import os

# Ensure matplotlib/fontconfig cache dirs are writable (important in restricted environments).
_RUNTIME_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runtime_cache")
os.makedirs(_RUNTIME_CACHE_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _RUNTIME_CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RUNTIME_CACHE_DIR, "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import argparse
import re
import sys
import importlib
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

COMPARISON_CSV = 'cluster_comparison.csv'
COMPARISON_REPORT = 'cluster_comparison_report.txt'
METADATA_CSV = 'embeddings_metadata.csv'
METADATA_WITH_CLUSTERS_CSV = 'metadata_with_clusters.csv'
CLUSTERS_CSV = 'clusters_with_labels.csv'

# Fixed output names
OVERVIEW_REPORT_TXT = 'overview_report.txt'
CLUSTER_SIZES_PNG = 'cluster_sizes.png'
CLUSTER_PROPORTIONS_PIE_PNG = 'cluster_proportions_pie.png'

CLUSTER_HEATMAP_PNG = 'cluster_heatmap.png'
CLUSTER_MIX_STACKED_BAR_PNG = 'cluster_mix_stacked_bar.png'
CLUSTER_COMPARISON_CHART_PNG = 'cluster_comparison_chart.png'  # only when 2 or 3 groups
CLUSTER_DIFFERENCES_CHART_PNG = 'cluster_differences_chart.png'  # only when 2 groups
CLUSTER_FACETS_PNG = 'cluster_distribution_facets.png'  # top 5 groups when many
CLUSTER_TREEMAP_HTML = 'cluster_treemap.html'
CLUSTER_SANKEY_HTML = 'cluster_group_sankey.html'


def get_categories_from_comparison(comparison_df):
    """Infer group category names from comparison CSV columns (*_Count)."""
    count_cols = [c for c in comparison_df.columns if c.endswith('_Count')]
    return [c.replace('_Count', '') for c in count_cols]


def get_group_col_from_report(report_path=COMPARISON_REPORT):
    """Parse group-by column name from step 3 report. Returns None if not found."""
    if not os.path.isfile(report_path):
        return None
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            text = f.read()
        m = re.search(r"CLUSTER COMPARISON BY\s+'([^']+)'", text)
        return m.group(1).strip() if m else None
    except Exception:
        return None


def parse_chi_square_from_report(report_path=COMPARISON_REPORT):
    """Parse chi-square statistic and p-value from step 3 report."""
    if not os.path.isfile(report_path):
        return None, None
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            text = f.read()
        m_stat = re.search(r"Chi-square statistic:\s*([0-9.]+)", text)
        m_p = re.search(r"P-value:\s*([0-9.]+)", text)
        chi2 = float(m_stat.group(1)) if m_stat else None
        p = float(m_p.group(1)) if m_p else None
        return chi2, p
    except Exception:
        return None, None


def load_fixed_mode_data():
    """Load all necessary data files for default (fixed outputs) mode."""
    print("Loading data...")
    comparison_df = pd.read_csv(COMPARISON_CSV)
    metadata = pd.read_csv(METADATA_CSV)
    clusters_df = pd.read_csv(CLUSTERS_CSV)
    metadata_with_clusters = pd.read_csv(METADATA_WITH_CLUSTERS_CSV)
    print("✓ Data loaded")
    return comparison_df, metadata, clusters_df, metadata_with_clusters


def create_cluster_comparison_chart(comparison_df, categories, output_file):
    """Create bar chart comparing clusters across groups (data-driven)."""
    if not categories:
        print("  Skip comparison chart: no group columns found")
        return
    if len(categories) not in (2, 3):
        print("  Skip comparison chart: only generated when there are 2 or 3 groups")
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


def _infer_group_totals_from_comparison(comparison_df, categories):
    totals = {}
    for cat in categories:
        col = f'{cat}_Count'
        if col in comparison_df.columns:
            totals[cat] = int(pd.to_numeric(comparison_df[col], errors='coerce').fillna(0).sum())
    return totals


def _cluster_totals_from_comparison(comparison_df, categories):
    count_cols = [f'{c}_Count' for c in categories if f'{c}_Count' in comparison_df.columns]
    if not count_cols:
        return pd.Series([0] * len(comparison_df), index=comparison_df.index)
    numeric = comparison_df[count_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    return numeric.sum(axis=1)


def create_cluster_sizes_chart(clusters_df, total_rows, output_file=CLUSTER_SIZES_PNG, max_bars=25):
    """Bar chart of cluster sizes (top N)."""
    df = clusters_df.sort_values('Message_Count', ascending=False).head(max_bars)
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.35)))
    y = np.arange(len(df))[::-1]
    ax.barh(y, df['Message_Count'], color='#2E86AB', alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df['Cluster_Label'], fontsize=9)
    ax.set_xlabel('Number of messages', fontsize=11, fontweight='bold')
    ax.set_title('Cluster sizes (top clusters by count)', fontsize=12, fontweight='bold', pad=12)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {output_file}")


def create_cluster_proportions_pie(clusters_df, total_rows, output_file=CLUSTER_PROPORTIONS_PIE_PNG, top_n=10):
    """Pie chart of cluster proportions (top N clusters + 'Other')."""
    df = clusters_df.sort_values('Message_Count', ascending=False).head(top_n)
    sizes = df['Message_Count'].tolist()
    labels = df['Cluster_Label'].tolist()
    other_count = int(total_rows) - int(sum(sizes))
    if other_count > 0 and len(clusters_df) > top_n:
        sizes.append(other_count)
        labels.append('Other')
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.75
    )
    for t in texts:
        t.set_fontsize(8)
    plt.setp(autotexts, size=8)
    ax.set_title('Cluster proportions (top + Other)', fontsize=12, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {output_file}")


def write_overview_report(
    clusters_df,
    total_rows,
    group_col,
    categories,
    group_totals,
    chi2=None,
    chi2_p=None,
    output_file=OVERVIEW_REPORT_TXT
):
    """Write an overview text report (overall + group counts)."""
    lines = []
    lines.append("=" * 60)
    lines.append("OVERVIEW SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total messages in clusters: {int(total_rows):,}")
    lines.append(f"Number of clusters: {len(clusters_df)}")
    if categories:
        lines.append(f"Group-by column: {group_col or '(unknown)'}")
        lines.append(f"Number of groups: {len(categories)}")
        lines.append("")
        lines.append("Messages per group:")
        for cat in categories:
            n = group_totals.get(cat, 0)
            pct = (n / total_rows * 100) if total_rows else 0
            lines.append(f"  {cat}: {n:,} ({pct:.1f}%)")
    if chi2 is not None and chi2_p is not None:
        lines.append("")
        lines.append("Step 3 chi-square (clusters × groups):")
        lines.append(f"  Chi-square statistic: {chi2:.4f}")
        lines.append(f"  P-value: {chi2_p:.6f}")

    lines.append("")
    lines.append("-" * 60)
    lines.append("Clusters (by size, largest first)")
    lines.append("-" * 60)
    clusters_sorted = clusters_df.sort_values('Message_Count', ascending=False).reset_index(drop=True)
    for _, row in clusters_sorted.iterrows():
        pct = (row['Message_Count'] / total_rows * 100) if total_rows else 0
        lines.append(f"  {row['Cluster_Label']}")
        lines.append(f"    Count: {int(row['Message_Count']):,} ({pct:.1f}%)")
        if pd.notna(row.get('Top_Keywords')) and str(row['Top_Keywords']).strip():
            lines.append(f"    Keywords: {row['Top_Keywords']}")
        lines.append("")

    lines.append("=" * 60)
    text = "\n".join(lines)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    print(f"✓ Saved to {output_file}")


def create_cluster_mix_stacked_bar(comparison_df, categories, output_file, top_clusters=10):
    """100% stacked bar: one bar per group, segments = top clusters + Other."""
    if not categories:
        print("  Skip stacked bar: no group columns found")
        return
    print("Creating stacked bar chart (100%)...")

    df = comparison_df.copy()
    df['Total_Count'] = _cluster_totals_from_comparison(df, categories)
    df = df.sort_values('Total_Count', ascending=False)
    top = df.head(top_clusters).copy()
    other = df.iloc[top_clusters:].copy()

    percent_cols = [f'{c}_Percent' for c in categories]
    if not all(c in df.columns for c in percent_cols):
        print("  Skip stacked bar: missing percent columns")
        return

    labels = top['Cluster_Label'].tolist() + (['Other'] if len(other) > 0 else [])
    values_by_group = {}
    for cat in categories:
        pcol = f'{cat}_Percent'
        vals = top[pcol].tolist()
        if len(other) > 0:
            other_pct = max(0.0, 100.0 - float(np.nansum(vals)))
            vals.append(other_pct)
        values_by_group[cat] = vals

    fig, ax = plt.subplots(figsize=(12, max(6, len(categories) * 0.4)))
    bottom = np.zeros(len(categories))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))

    for i, lbl in enumerate(labels):
        heights = [values_by_group[cat][i] for cat in categories]
        ax.bar(categories, heights, bottom=bottom, label=lbl, color=colors[i], alpha=0.9)
        bottom += np.array(heights)

    ax.set_ylabel('Percentage of messages (%)', fontsize=11, fontweight='bold')
    ax.set_title('Cluster mix within each group (top clusters + Other)', fontsize=12, fontweight='bold', pad=12)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {output_file}")


def create_cluster_facets(comparison_df, categories, output_file, max_groups=5, top_clusters_per_group=12):
    """Facet bar charts: one subplot per group; cap groups to top N totals when many."""
    if not categories:
        print("  Skip facets: no group columns found")
        return

    totals = _infer_group_totals_from_comparison(comparison_df, categories)
    if len(categories) > max_groups:
        cats = sorted(categories, key=lambda c: totals.get(c, 0), reverse=True)[:max_groups]
        print(f"Creating facets (top {max_groups} groups by total count): {cats}")
    else:
        cats = categories
        print(f"Creating facets: {cats}")

    n = len(cats)
    ncols = 2
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, max(6, nrows * 4)))
    axes = np.array(axes).reshape(-1)

    for i, cat in enumerate(cats):
        ax = axes[i]
        pct_col = f'{cat}_Percent'
        if pct_col not in comparison_df.columns:
            ax.set_axis_off()
            continue
        top = comparison_df.nlargest(top_clusters_per_group, pct_col)[['Cluster_Label', pct_col]].copy()
        top = top.sort_values(pct_col, ascending=True)
        ax.barh(top['Cluster_Label'], top[pct_col], color='#2E86AB', alpha=0.85)
        ax.set_title(str(cat), fontsize=12, fontweight='bold')
        ax.set_xlabel('% of messages', fontsize=10)
        ax.grid(axis='x', alpha=0.25)
        ax.tick_params(axis='y', labelsize=8)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()

    fig.suptitle('Cluster distribution by group (facets)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=250, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {output_file}")


def create_treemap_html(comparison_df, categories, output_file=CLUSTER_TREEMAP_HTML):
    """Treemap (interactive HTML): cluster sizes colored by dominant group (or 2-group difference)."""
    print("Creating treemap (HTML)...")
    try:
        px = importlib.import_module("plotly.express")
    except Exception as e:
        print("  Skipping treemap (plotly import failed):", e)
        print("  Python used:", sys.executable)
        return False

    df = comparison_df.copy()
    df['Total_Count'] = _cluster_totals_from_comparison(df, categories)
    percent_cols = [f'{c}_Percent' for c in categories if f'{c}_Percent' in df.columns]

    if len(categories) == 2 and 'Difference' in df.columns:
        color_col = 'Difference'
        fig = px.treemap(
            df,
            path=[px.Constant("Clusters"), 'Cluster_Label'],
            values='Total_Count',
            color=color_col,
            color_continuous_scale='RdBu',
            hover_data=['Cluster_ID'] + percent_cols
        )
    else:
        # dominant group by percent
        if percent_cols:
            df['Dominant_Group'] = df[percent_cols].idxmax(axis=1).str.replace('_Percent', '', regex=False)
            df['Dominant_Share'] = df[percent_cols].max(axis=1)
        else:
            df['Dominant_Group'] = 'N/A'
            df['Dominant_Share'] = np.nan
        fig = px.treemap(
            df,
            path=[px.Constant("Clusters"), 'Cluster_Label'],
            values='Total_Count',
            color='Dominant_Group',
            hover_data=['Cluster_ID', 'Dominant_Share'] + percent_cols
        )
    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), title="Cluster treemap")
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"✓ Saved to {output_file}")
    return True


def create_sankey_html(comparison_df, categories, output_file=CLUSTER_SANKEY_HTML, top_clusters=20):
    """Sankey (interactive HTML): clusters -> groups with link values = counts."""
    print("Creating Sankey diagram (HTML)...")
    try:
        go = importlib.import_module("plotly.graph_objects")
    except Exception as e:
        print("  Skipping Sankey (plotly import failed):", e)
        print("  Python used:", sys.executable)
        return False

    df = comparison_df.copy()
    df['Total_Count'] = _cluster_totals_from_comparison(df, categories)
    df = df.sort_values('Total_Count', ascending=False).head(top_clusters)

    cluster_labels = df['Cluster_Label'].astype(str).tolist()
    group_labels = [str(c) for c in categories]
    labels = cluster_labels + group_labels

    cluster_index = {lbl: i for i, lbl in enumerate(cluster_labels)}
    group_index = {lbl: len(cluster_labels) + i for i, lbl in enumerate(group_labels)}

    sources, targets, values = [], [], []
    for _, row in df.iterrows():
        c_lbl = str(row['Cluster_Label'])
        for g in group_labels:
            col = f'{g}_Count'
            if col not in df.columns:
                continue
            v = int(pd.to_numeric(row.get(col, 0), errors='coerce') or 0)
            if v <= 0:
                continue
            sources.append(cluster_index[c_lbl])
            targets.append(group_index[g])
            values.append(v)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=15,
                    thickness=15,
                    line=dict(color="black", width=0.2),
                    label=labels
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values
                )
            )
        ]
    )
    fig.update_layout(title_text=f"Cluster → Group Sankey (top {top_clusters} clusters)", font_size=10)
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"✓ Saved to {output_file}")
    return True


def llm_mode_interactive(confirm_before_run=False):
    """Interactive LLM mode: prompt user and execute generated code (no fixed outputs)."""
    # Load dataframes for user convenience (same filenames as fixed mode)
    dfs = {}
    for name, path in [
        ("comparison_df", COMPARISON_CSV),
        ("clusters_df", CLUSTERS_CSV),
        ("metadata_df", METADATA_CSV),
        ("metadata_with_clusters_df", METADATA_WITH_CLUSTERS_CSV),
    ]:
        if os.path.isfile(path):
            dfs[name] = pd.read_csv(path)
    if not dfs:
        raise FileNotFoundError("No input CSVs found. Run steps 02/03 first (and keep outputs in this folder).")

    # Gemini setup (reuse step 02 API key)
    try:
        from dotenv import load_dotenv
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError("LLM mode requires python-dotenv and google-generativeai. Install with: pip install python-dotenv google-generativeai") from e

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found. Put it in .env as GOOGLE_API_KEY=... (same as step 02).")
    genai.configure(api_key=api_key)

    # Optional plotly availability for generated code
    try:
        px = importlib.import_module("plotly.express")
        go = importlib.import_module("plotly.graph_objects")
    except Exception:
        px = None
        go = None

    def build_context():
        parts = []
        parts.append("You are generating Python code only. No markdown fences. No explanations.")
        parts.append("Available pandas DataFrames are already loaded in memory:")
        for df_name, df in dfs.items():
            cols = list(df.columns)
            parts.append(f"- {df_name}: {len(df):,} rows; columns={cols}")
        parts.append("Rules:")
        parts.append("- Use pandas/matplotlib/seaborn/plotly to compute or plot as requested.")
        parts.append("- If you create a matplotlib plot, save it with plt.savefig('llm_output_1.png', dpi=200, bbox_inches='tight') (or increment the number).")
        parts.append("- If you create a plotly figure, save it with fig.write_html('llm_output_1.html', include_plotlyjs='cdn').")
        parts.append("- Print key results to stdout.")
        parts.append("- Do not read/write any files except the output charts you create.")
        return "\n".join(parts)

    system_prompt = build_context()
    model_names = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-2.5-pro']

    def strip_code_fences(text):
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\n", "", t)
            t = re.sub(r"\n```$", "", t)
        return t.strip()

    import builtins
    safe_builtins = {
        "print": print,
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "sorted": sorted,
        "list": list,
        "dict": dict,
        "set": set,
        "float": float,
        "int": int,
        "str": str,
        "enumerate": enumerate,
        "zip": zip,
        "__import__": builtins.__import__,
    }

    exec_globals = {
        "__builtins__": safe_builtins,
        "pd": pd,
        "np": np,
        "plt": plt,
        "sns": sns,
        "WordCloud": WordCloud,
        "os": os,
        "re": re,
    }
    if px is not None:
        exec_globals["px"] = px
    if go is not None:
        exec_globals["go"] = go
    exec_globals.update(dfs)

    print("\n" + "=" * 60)
    print("LLM MODE")
    print("=" * 60)
    print("What kind of numbers or visualization do you want?")

    counter = 1
    while True:
        user_req = input("\nInput your request (or 'done' to exit): ").strip()
        if not user_req:
            continue
        if user_req.lower() in {"done", "exit", "quit"}:
            break

        prompt = f"{system_prompt}\n\nUser request:\n{user_req}\n\nReturn only executable Python code."
        code = None
        last_err = None
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                resp = model.generate_content(prompt)
                code = strip_code_fences(getattr(resp, "text", "") or "")
                if code:
                    break
            except Exception as e:
                last_err = e
                continue
        if not code:
            raise RuntimeError(f"LLM did not return code. Last error: {last_err}")

        print("\n--- Generated code ---\n")
        print(code)
        print("\n--- End code ---\n")
        if confirm_before_run:
            run = input("Run this? [y/N] ").strip().lower()
            if run not in {"y", "yes"}:
                continue

        # Encourage unique output names by providing a counter variable
        exec_globals["LLM_OUTPUT_INDEX"] = counter
        try:
            exec(code, exec_globals, {})
        except Exception as e:
            print(f"\nError while running generated code: {e}")
        counter += 1


def _plotly_available():
    """Return True if plotly can be imported (for treemap/Sankey)."""
    try:
        importlib.import_module("plotly.express")
        return True
    except Exception:
        return False


def fixed_outputs_mode():
    """Default mode: generate the fixed set of outputs."""
    if not _plotly_available():
        print("\nPlotly not available for this Python. Treemap and Sankey will be skipped.")
        print("Python used by this script:", sys.executable)
        print("To enable: run in terminal:", sys.executable, "-m pip install plotly")
        print()

    comparison_df, metadata, clusters_df, metadata_with_clusters = load_fixed_mode_data()
    categories = get_categories_from_comparison(comparison_df)
    group_col = get_group_col_from_report()
    chi2, chi2_p = parse_chi_square_from_report()

    # Totals and overall counts
    group_totals = _infer_group_totals_from_comparison(comparison_df, categories)
    if 'Cluster' in metadata_with_clusters.columns:
        total_rows = int((pd.to_numeric(metadata_with_clusters['Cluster'], errors='coerce').fillna(-1) >= 0).sum())
    else:
        total_rows = int(pd.to_numeric(clusters_df.get('Message_Count', 0), errors='coerce').fillna(0).sum())

    print("\n" + "=" * 60)
    print("CREATING REPORTS AND VISUALIZATIONS (DEFAULT MODE)")
    print("=" * 60)
    print(f"Groups from comparison: {categories}")
    print(f"Group-by column (for word clouds): {group_col or 'not found (skip word clouds)'}")

    generated = []

    # Overall outputs
    write_overview_report(
        clusters_df=clusters_df,
        total_rows=total_rows,
        group_col=group_col,
        categories=categories,
        group_totals=group_totals,
        chi2=chi2,
        chi2_p=chi2_p,
        output_file=OVERVIEW_REPORT_TXT
    )
    generated.append(OVERVIEW_REPORT_TXT)
    create_cluster_sizes_chart(clusters_df, total_rows, CLUSTER_SIZES_PNG)
    generated.append(CLUSTER_SIZES_PNG)
    create_cluster_proportions_pie(clusters_df, total_rows, CLUSTER_PROPORTIONS_PIE_PNG, top_n=10)
    generated.append(CLUSTER_PROPORTIONS_PIE_PNG)

    # Comparison outputs
    create_cluster_heatmap(comparison_df, categories, CLUSTER_HEATMAP_PNG)
    generated.append(CLUSTER_HEATMAP_PNG)

    create_cluster_mix_stacked_bar(comparison_df, categories, CLUSTER_MIX_STACKED_BAR_PNG, top_clusters=10)
    generated.append(CLUSTER_MIX_STACKED_BAR_PNG)

    if group_col and group_col in metadata.columns:
        for cat in categories:
            safe = re.sub(r'[^\w\-]', '_', str(cat))
            out = f'wordcloud_{safe}.png'
            create_wordcloud(metadata, group_col, cat, out)
            generated.append(out)
    else:
        print("  Skipping word clouds (no group column in report or metadata)")

    create_cluster_facets(comparison_df, categories, CLUSTER_FACETS_PNG, max_groups=5, top_clusters_per_group=12)
    generated.append(CLUSTER_FACETS_PNG)

    if create_treemap_html(comparison_df, categories, CLUSTER_TREEMAP_HTML):
        generated.append(CLUSTER_TREEMAP_HTML)
    if create_sankey_html(comparison_df, categories, CLUSTER_SANKEY_HTML, top_clusters=20):
        generated.append(CLUSTER_SANKEY_HTML)

    create_cluster_comparison_chart(comparison_df, categories, CLUSTER_COMPARISON_CHART_PNG)
    if len(categories) in (2, 3):
        generated.append(CLUSTER_COMPARISON_CHART_PNG)

    create_difference_chart(comparison_df, categories, CLUSTER_DIFFERENCES_CHART_PNG)
    if 'Difference' in comparison_df.columns and 'P_Value' in comparison_df.columns:
        generated.append(CLUSTER_DIFFERENCES_CHART_PNG)

    print("\n" + "=" * 60)
    print("Step 4 Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    for f in generated:
        print(f"  - {f}")
    print("=" * 60)


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Step 4: Create cluster reports and visualizations (default), or run interactive LLM mode (--llm)."
        )
        parser.add_argument(
            "--llm",
            action="store_true",
            help="Interactive LLM mode (Gemini): prompt for a custom request and run generated code. Skips fixed outputs."
        )
        parser.add_argument(
            "--confirm",
            action="store_true",
            help="In LLM mode: ask 'Run this? [y/N]' before executing generated code (default: run automatically)."
        )
        args = parser.parse_args()

        if args.llm:
            llm_mode_interactive(confirm_before_run=args.confirm)
        else:
            fixed_outputs_mode()
    except FileNotFoundError as e:
        print(f"Error: {e} not found. Please run previous steps first.")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
