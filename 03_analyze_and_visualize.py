#!/usr/bin/env python3
"""
Step 3: Analyze, compare clusters and create reports/visualizations.
Compares clusters across a grouping column (chi-square, z-tests when 2 groups),
then produces overview report and all comparison charts. Optional: --compare-only (no charts),
or --llm for interactive ad-hoc numbers/visualizations.
"""

import os

_RUNTIME_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".runtime_cache")
os.makedirs(_RUNTIME_CACHE_DIR, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", _RUNTIME_CACHE_DIR)
os.environ.setdefault("MPLCONFIGDIR", os.path.join(_RUNTIME_CACHE_DIR, "matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import argparse
import json
import re
import sys
import importlib
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

METADATA_CSV = 'embeddings_metadata.csv'
METADATA_WITH_CLUSTERS_CSV = 'metadata_with_clusters.csv'
CLUSTERS_CSV = 'clusters_with_labels.csv'


def _load_visualization_config():
    """Load visualization config from config/config.json."""
    config_path = Path(__file__).resolve().parent / "config" / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    viz = raw["visualization"]
    if not isinstance(viz, dict) or "output_files" not in viz:
        raise ValueError(
            "config/config.json must have a 'visualization' object with 'output_files'. "
            f"Got: {type(viz).__name__}"
        )
    return viz


_VIZ = _load_visualization_config()
_of = _VIZ["output_files"]
if _VIZ.get("chart") and _VIZ["chart"].get("font_family") == "sans-serif":
    _VIZ["chart"] = {**_VIZ["chart"], "font_family": "DejaVu Sans"}
TEXT_COLUMN = 'text'


def _text_column(metadata_df):
    """Text column name: TEXT_COLUMN or 'Message'."""
    if metadata_df is None or metadata_df.columns is None:
        return TEXT_COLUMN
    if TEXT_COLUMN in metadata_df.columns:
        return TEXT_COLUMN
    if 'Message' in metadata_df.columns:
        return 'Message'
    return TEXT_COLUMN


EXCLUDE_FROM_CATEGORY = {TEXT_COLUMN, 'Message', 'Cluster', 'Cluster_ID', 'Cluster_Label'}
MAX_CARDINALITY_FOR_CATEGORY = 50


def detect_category_columns(metadata_with_clusters):
    """Column names that look like categories (limited cardinality)."""
    candidates = []
    for col in metadata_with_clusters.columns:
        if col in EXCLUDE_FROM_CATEGORY:
            continue
        s = metadata_with_clusters[col].dropna().astype(str).str.strip()
        s = s[s != ""]
        if len(s) == 0:
            continue
        n_unique = s.nunique()
        if n_unique < 2:
            continue
        if n_unique <= MAX_CARDINALITY_FOR_CATEGORY:
            candidates.append(col)
    return candidates


def _normalize_clusters_df(clusters_df):
    """Ensure Cluster_ID and Cluster_Label (rename Topic_* if present)."""
    if clusters_df is None or len(clusters_df) == 0:
        return clusters_df
    if 'Cluster_Label' not in clusters_df.columns and 'Topic_Label' in clusters_df.columns:
        clusters_df = clusters_df.rename(columns={'Topic_Label': 'Cluster_Label'})
    if 'Cluster_ID' not in clusters_df.columns and 'Topic_ID' in clusters_df.columns:
        clusters_df = clusters_df.rename(columns={'Topic_ID': 'Cluster_ID'})
    return clusters_df


def load_data(clusters_file, metadata_file):
    print("Loading data...")
    clusters_df = pd.read_csv(clusters_file)
    clusters_df = _normalize_clusters_df(clusters_df)
    metadata = pd.read_csv(metadata_file)
    print(f"✓ Loaded {len(clusters_df)} clusters")
    print(f"✓ Loaded {len(metadata):,} rows")
    return clusters_df, metadata


def calculate_proportions(metadata_with_clusters, clusters_df, group_col, categories):
    totals = {cat: len(metadata_with_clusters[metadata_with_clusters[group_col].astype(str) == str(cat)]) for cat in categories}
    total_all = sum(totals.values())
    print(f"\nGroup column: '{group_col}'")
    for cat in categories:
        print(f"  Total rows - {cat}: {totals[cat]:,}")
    if total_all == 0:
        raise ValueError(f"No rows in any of the groups: {categories}")

    if len(clusters_df) == 0:
        print("  Warning: No clusters in clusters_with_labels.csv (e.g. all noise in step 02). Comparison will be empty.")
        cols = ['Cluster_Label', 'Cluster_ID'] + [f'{c}_Count' for c in categories] + [f'{c}_Percent' for c in categories]
        if len(categories) == 2:
            cols.append('Difference')
        return pd.DataFrame(columns=cols), totals

    comparison_data = []
    for _, cluster_row in clusters_df.iterrows():
        cluster_id = cluster_row['Cluster_ID']
        cluster_data = metadata_with_clusters[metadata_with_clusters['Cluster'] == cluster_id]

        row = {
            'Cluster_Label': cluster_row['Cluster_Label'],
            'Cluster_ID': cluster_id,
        }
        counts = {}
        percents = {}
        for cat in categories:
            count = len(cluster_data[cluster_data[group_col].astype(str) == str(cat)])
            counts[cat] = count
            percents[cat] = (count / totals[cat] * 100) if totals[cat] > 0 else 0
            row[f'{cat}_Count'] = count
            row[f'{cat}_Percent'] = percents[cat]

        if len(categories) == 2:
            row['Difference'] = percents[categories[0]] - percents[categories[1]]
        comparison_data.append(row)

    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = _ensure_comparison_numeric(comparison_df)
    return comparison_df, totals


def perform_z_test(n1, x1, n2, x2, alpha=0.05):
    """Z-test for two proportions. Returns z_score, p_value, (ci_lower, ci_upper) in pct points."""
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan, (np.nan, np.nan)
    p1 = x1 / n1
    p2 = x2 / n2
    p_pooled = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    if se == 0:
        return np.nan, np.nan, (np.nan, np.nan)
    z_score = (p1 - p2) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    z_critical = stats.norm.ppf(1 - alpha/2)
    margin_error = z_critical * se
    ci_lower = (p1 - p2 - margin_error) * 100
    ci_upper = (p1 - p2 + margin_error) * 100
    return z_score, p_value, (ci_lower, ci_upper)


def perform_chi_square_test(metadata_with_clusters, clusters_df, group_col):
    """Chi-square test (clusters × groups); returns standardized residuals for 3+ groups."""
    print("\nPerforming chi-square test for overall association...")
    contingency = pd.crosstab(
        metadata_with_clusters['Cluster'],
        metadata_with_clusters[group_col].astype(str)
    )
    chi2, p_value, dof, expected = chi2_contingency(contingency)
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Degrees of freedom: {dof}")
    if p_value < 0.05:
        print(f"✓ Result: Groups have significantly different cluster distributions (p < 0.05)")
    else:
        print("✗ Result: No significant difference in cluster distributions (p >= 0.05)")
    expected_df = pd.DataFrame(expected, index=contingency.index, columns=contingency.columns)
    with np.errstate(divide='ignore', invalid='ignore'):
        residuals_df = (contingency - expected_df) / np.sqrt(expected_df)
    residuals_df = residuals_df.replace([np.inf, -np.inf], np.nan)
    return chi2, p_value, dof, residuals_df


def add_statistical_tests(comparison_df, totals, categories):
    """Add z-tests and CIs for 2 groups only."""
    if len(categories) != 2:
        print("\nSkipping z-tests (only run for 2 groups).")
        comparison_df['Z_Score'] = np.nan
        comparison_df['P_Value'] = np.nan
        comparison_df['CI_Lower'] = np.nan
        comparison_df['CI_Upper'] = np.nan
        return comparison_df

    print("\nPerforming z-tests for each cluster (2 groups)...")
    n1, n2 = totals[categories[0]], totals[categories[1]]
    z_scores, p_values, ci_lowers, ci_uppers = [], [], [], []
    for _, row in comparison_df.iterrows():
        x1, x2 = row[f'{categories[0]}_Count'], row[f'{categories[1]}_Count']
        z, p, (ci_lo, ci_hi) = perform_z_test(n1, x1, n2, x2)
        z_scores.append(z)
        p_values.append(p)
        ci_lowers.append(ci_lo)
        ci_uppers.append(ci_hi)
    comparison_df['Z_Score'] = z_scores
    comparison_df['P_Value'] = p_values
    comparison_df['CI_Lower'] = ci_lowers
    comparison_df['CI_Upper'] = ci_uppers
    print("✓ Statistical tests complete")
    return comparison_df


def generate_report(comparison_df, chi2, chi2_p, totals, categories, group_col, residuals_df=None):
    report_lines = []
    report_lines.append("="*60)
    report_lines.append(f"CLUSTER COMPARISON BY '{group_col}'")
    report_lines.append("="*60)
    report_lines.append(f"\nTotal rows per group:")
    for cat in categories:
        report_lines.append(f"  {cat}: {totals[cat]:,}")
    report_lines.append(f"\nChi-Square Test Results:")
    report_lines.append(f"  Chi-square statistic: {chi2:.4f}")
    report_lines.append(f"  P-value: {chi2_p:.6f}")
    report_lines.append(f"  Conclusion: {'Significant difference' if chi2_p < 0.05 else 'No significant difference'}")
    if len(categories) >= 3:
        report_lines.append("  Note: For 3+ groups, only the overall chi-square test is performed (no per-cluster z-tests).")

    report_lines.append(f"\n\nTOP CLUSTERS BY GROUP:")
    for cat in categories:
        pct_col = f'{cat}_Percent'
        if pct_col not in comparison_df.columns:
            continue
        report_lines.append(f"\nHighest share in {cat}:")
        top = comparison_df.nlargest(5, pct_col)
        for _, row in top.iterrows():
            pval = row.get('P_Value', np.nan)
            report_lines.append(f"  {row['Cluster_Label']}: {row[pct_col]:.2f}%"
                + (f" (p={pval:.4f})" if not np.isnan(pval) else ""))

    if len(categories) == 2 and 'Difference' in comparison_df.columns and 'P_Value' in comparison_df.columns:
        report_lines.append(f"\n\nSIGNIFICANT DIFFERENCES (p < 0.05):")
        significant = comparison_df[comparison_df['P_Value'] < 0.05].copy()
        significant = significant.sort_values('Difference', key=abs, ascending=False)
        if len(significant) > 0:
            for _, row in significant.head(10).iterrows():
                d = row['Difference']
                direction = f"more in {categories[0]}" if d > 0 else f"more in {categories[1]}"
                report_lines.append(f"  {row['Cluster_Label']}: {abs(d):.2f}% {direction} "
                    f"(CI: {row['CI_Lower']:.2f}% to {row['CI_Upper']:.2f}%)")
        else:
            report_lines.append("  No significant differences found")
    elif len(categories) >= 3 and residuals_df is not None and isinstance(residuals_df, pd.DataFrame):
        report_lines.append(f"\n\nCELLS WITH LARGEST STANDARDIZED RESIDUALS (interpretation aid):")
        id_to_label = dict(zip(comparison_df['Cluster_ID'], comparison_df['Cluster_Label']))
        flat = residuals_df.stack().reset_index()
        flat.columns = ['Cluster_ID', 'Group', 'Std_Residual']
        flat['Abs'] = flat['Std_Residual'].abs()
        top = flat.sort_values('Abs', ascending=False).head(10)
        if len(top) == 0:
            report_lines.append("  (No residuals available)")
        else:
            for _, r in top.iterrows():
                cid = r['Cluster_ID']
                label = id_to_label.get(cid, f"Cluster_ID {cid}")
                direction = "over" if r['Std_Residual'] > 0 else "under"
                report_lines.append(f"  {label} (ID {cid}) in {r['Group']}: {r['Std_Residual']:.2f} ({direction}-represented vs expected)")

    report_lines.append("\n" + "="*60)
    return "\n".join(report_lines)


def get_categories_from_comparison(comparison_df):
    """Infer category names from *_Count columns."""
    count_cols = [c for c in comparison_df.columns if c.endswith('_Count')]
    return [c.replace('_Count', '') for c in count_cols]


def get_group_col_from_report(report_path=None):
    """Parse group-by column from comparison report."""
    report_path = report_path or _of["comparison_report"]
    if not os.path.isfile(report_path) or report_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.html')):
        return None
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            text = f.read()
        m = re.search(r"CLUSTER COMPARISON BY\s+'([^']+)'", text)
        return m.group(1).strip() if m else None
    except Exception:
        return None


def parse_chi_square_from_report(report_path=None):
    report_path = report_path or _of["comparison_report"]
    if not os.path.isfile(report_path) or report_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.html')):
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


def _ensure_comparison_numeric(comparison_df):
    """Coerce *_Count, *_Percent, Difference to numeric."""
    out = comparison_df.copy()
    for col in out.columns:
        if col.endswith('_Count') or col.endswith('_Percent') or col == 'Difference':
            out[col] = pd.to_numeric(out[col], errors='coerce')
    return out


def load_fixed_mode_data():
    """Load comparison CSV, report, metadata, clusters for fixed-output visualizations."""
    print("Loading data for visualizations...")
    comparison_df = pd.read_csv(_of["comparison_csv"])
    comparison_df = _ensure_comparison_numeric(comparison_df)
    metadata = pd.read_csv(METADATA_CSV)
    clusters_df = pd.read_csv(CLUSTERS_CSV)
    clusters_df = _normalize_clusters_df(clusters_df)
    metadata_with_clusters = pd.read_csv(METADATA_WITH_CLUSTERS_CSV)
    print("✓ Data loaded")
    return comparison_df, metadata, clusters_df, metadata_with_clusters


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


def create_cluster_comparison_chart(comparison_df, categories, output_file):
    """Bar chart: clusters across groups (2 or 3 groups)."""
    if not categories:
        return
    if len(categories) not in (2, 3):
        return
    print("Creating cluster comparison chart...")
    count_cols = [f'{c}_Count' for c in categories]
    df = comparison_df.copy()
    df['Total_Count'] = df[count_cols].sum(axis=1)
    top_clusters = df.nlargest(15, 'Total_Count')
    percent_cols = [f'{c}_Percent' for c in categories]
    if not all(c in top_clusters.columns for c in percent_cols):
        return
    fig, ax = plt.subplots(figsize=(14, 8))
    x = np.arange(len(top_clusters))
    n_groups = len(categories)
    width = 0.8 / n_groups
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_groups, 2)))
    for i, (cat, pct_col) in enumerate(zip(categories, percent_cols)):
        offset = (i - (n_groups - 1) / 2) * width
        ax.bar(x + offset, top_clusters[pct_col], width, label=cat, color=colors[i], alpha=0.8)
    fs_axis = _VIZ["chart"]["font_size_axis"]
    fs_title = _VIZ["chart"]["font_size_title"]
    ax.set_xlabel('Clusters', fontsize=fs_axis, fontweight='bold', fontfamily=_VIZ["chart"]["font_family"])
    ax.set_ylabel('Percentage (%)', fontsize=fs_axis, fontweight='bold', fontfamily=_VIZ["chart"]["font_family"])
    ax.set_title(f"{_VIZ['titles']['cluster_distribution_by_group']} ({', '.join(categories)})", fontsize=fs_title, fontweight='bold', pad=20, fontfamily=_VIZ["chart"]["font_family"])
    ax.set_xticks(x)
    fs_tick = _VIZ["chart"]["font_size_tick"]
    ax.set_xticklabels(top_clusters['Cluster_Label'], rotation=45, ha='right', fontsize=fs_tick, fontfamily=_VIZ["chart"]["font_family"])
    ax.legend(fontsize=_VIZ["chart"]["font_size_legend"], prop={"family": _VIZ["chart"]["font_family"]})
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=_VIZ["chart"]["dpi"], bbox_inches='tight')
    print(f"✓ Saved to {output_file}")
    plt.close()


def create_wordcloud(metadata, group_col, channel, output_file):
    print(f"Creating word cloud for {channel}...")
    col = metadata.get(group_col)
    if col is None:
        return
    text_col = _text_column(metadata)
    if text_col not in metadata.columns:
        return
    channel_texts = metadata[metadata[group_col].astype(str).str.strip() == str(channel)][text_col].dropna().astype(str).tolist()
    if not channel_texts:
        return
    text = ' '.join(channel_texts)
    wc_bg = _VIZ["chart"]["wordcloud_background"]
    wc_cmap = _VIZ["chart"]["wordcloud_colormap"]
    wordcloud = WordCloud(width=1200, height=600, background_color=wc_bg, max_words=100, colormap=wc_cmap, relative_scaling=0.5, collocations=False).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{_VIZ['titles']['wordcloud']}: {channel}", fontsize=_VIZ["chart"]["font_size_title"] + 2, fontweight='bold', pad=20, fontfamily=_VIZ["chart"]["font_family"])
    plt.tight_layout()
    plt.savefig(output_file, dpi=_VIZ["chart"]["dpi"], bbox_inches='tight')
    print(f"✓ Saved to {output_file}")
    plt.close()


def create_cluster_heatmap(comparison_df, categories, output_file):
    if not categories:
        return
    print("Creating cluster heatmap...")
    count_cols = [f'{c}_Count' for c in categories]
    df = comparison_df.copy()
    df['Total_Count'] = df[count_cols].sum(axis=1)
    top_clusters = df.nlargest(20, 'Total_Count')
    percent_cols = [f'{c}_Percent' for c in categories]
    if not all(c in top_clusters.columns for c in percent_cols):
        return
    heatmap_data = top_clusters[['Cluster_Label'] + percent_cols].copy()
    heatmap_data = heatmap_data.set_index('Cluster_Label')
    heatmap_data.columns = categories
    fig, ax = plt.subplots(figsize=(10, 12))
    hm_cmap = _VIZ["chart"]["heatmap_cmap"]
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap=hm_cmap, cbar_kws={'label': 'Percentage (%)'}, ax=ax, linewidths=0.5, linecolor='gray')
    fs_axis = _VIZ["chart"]["font_size_axis"]
    fs_title = _VIZ["chart"]["font_size_title"]
    ax.set_title(f"{_VIZ['titles']['cluster_distribution_by_group']} ({', '.join(categories)})", fontsize=fs_title, fontweight='bold', pad=20, fontfamily=_VIZ["chart"]["font_family"])
    ax.set_xlabel('Group', fontsize=fs_axis, fontweight='bold', fontfamily=_VIZ["chart"]["font_family"])
    ax.set_ylabel('Clusters', fontsize=fs_axis, fontweight='bold', fontfamily=_VIZ["chart"]["font_family"])
    plt.tight_layout()
    plt.savefig(output_file, dpi=_VIZ["chart"]["dpi"], bbox_inches='tight')
    print(f"✓ Saved to {output_file}")
    plt.close()


def create_difference_chart(comparison_df, categories, output_file):
    if 'Difference' not in comparison_df.columns or 'P_Value' not in comparison_df.columns:
        return
    print("Creating difference chart...")
    significant = comparison_df[comparison_df['P_Value'] < 0.05].copy()
    significant = significant.sort_values('Difference', key=abs, ascending=False).head(15)
    if len(significant) == 0:
        return
    diff_label = f"{categories[0]} - {categories[1]}" if len(categories) == 2 else "Difference"
    fig, ax = plt.subplots(figsize=(12, 8))
    c_pos = _VIZ["chart"]["difference_positive_color"]
    c_neg = _VIZ["chart"]["difference_negative_color"]
    colors = [c_pos if d > 0 else c_neg for d in significant['Difference']]
    ax.barh(significant['Cluster_Label'], significant['Difference'], color=colors, alpha=0.8)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    fs_axis = _VIZ["chart"]["font_size_axis"]
    fs_title = _VIZ["chart"]["font_size_title"]
    ax.set_xlabel(f'Difference in Percentage ({diff_label})', fontsize=fs_axis, fontweight='bold', fontfamily=_VIZ["chart"]["font_family"])
    ax.set_ylabel('Clusters', fontsize=fs_axis, fontweight='bold', fontfamily=_VIZ["chart"]["font_family"])
    ax.set_title(_VIZ["titles"]["significant_differences"], fontsize=fs_title, fontweight='bold', pad=20, fontfamily=_VIZ["chart"]["font_family"])
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=_VIZ["chart"]["dpi"], bbox_inches='tight')
    print(f"✓ Saved to {output_file}")
    plt.close()


def create_cluster_sizes_chart(clusters_df, total_rows, output_file, max_bars=25):
    df = clusters_df.sort_values('Message_Count', ascending=False).head(max_bars)
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.35)))
    y = np.arange(len(df))[::-1]
    bar_color = _VIZ["chart"]["bar_color"]
    ax.barh(y, df['Message_Count'], color=bar_color, alpha=0.85)
    ax.set_yticks(y)
    fs_tick = _VIZ["chart"]["font_size_tick"]
    ax.set_yticklabels(df['Cluster_Label'], fontsize=fs_tick, fontfamily=_VIZ["chart"]["font_family"])
    ax.set_xlabel('Number of rows', fontsize=_VIZ["chart"]["font_size_axis"] - 1, fontweight='bold', fontfamily=_VIZ["chart"]["font_family"])
    ax.set_title(_VIZ["titles"]["cluster_sizes"], fontsize=_VIZ["chart"]["font_size_axis"], fontweight='bold', pad=12, fontfamily=_VIZ["chart"]["font_family"])
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_file, dpi=_VIZ["chart"]["dpi_small"], bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {output_file}")


def create_cluster_proportions_pie(clusters_df, total_rows, output_file, top_n=10):
    df = clusters_df.sort_values('Message_Count', ascending=False).head(top_n)
    sizes = df['Message_Count'].tolist()
    labels = df['Cluster_Label'].tolist()
    other_count = int(total_rows) - int(sum(sizes))
    if other_count > 0 and len(clusters_df) > top_n:
        sizes.append(other_count)
        labels.append('Other')
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(sizes)))
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, pctdistance=0.75)
    fs_tick = _VIZ["chart"]["font_size_tick"]
    for t in texts:
        t.set_fontsize(max(7, fs_tick - 1))
    plt.setp(autotexts, size=max(7, fs_tick - 1))
    ax.set_title(_VIZ["titles"]["cluster_proportions_pie"], fontsize=_VIZ["chart"]["font_size_axis"], fontweight='bold', pad=12, fontfamily=_VIZ["chart"]["font_family"])
    plt.tight_layout()
    plt.savefig(output_file, dpi=_VIZ["chart"]["dpi_small"], bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {output_file}")


def write_overview_report(clusters_df, total_rows, group_col, categories, group_totals, chi2=None, chi2_p=None, output_file=None):
    output_file = output_file or _of["overview_report"]
    lines = []
    lines.append("=" * 60)
    lines.append("OVERVIEW SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total rows in clusters: {int(total_rows):,}")
    lines.append(f"Number of clusters: {len(clusters_df)}")
    if categories:
        lines.append(f"Group-by column: {group_col or '(unknown)'}")
        lines.append(f"Number of groups: {len(categories)}")
        lines.append("")
        lines.append("Rows per group:")
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
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"✓ Saved to {output_file}")


def create_cluster_mix_stacked_bar(comparison_df, categories, output_file, top_clusters=10):
    if not categories:
        return
    print("Creating stacked bar chart (100%)...")
    df = comparison_df.copy()
    df['Total_Count'] = _cluster_totals_from_comparison(df, categories)
    df = df.sort_values('Total_Count', ascending=False)
    top = df.head(top_clusters).copy()
    other = df.iloc[top_clusters:].copy()
    percent_cols = [f'{c}_Percent' for c in categories]
    if not all(c in df.columns for c in percent_cols):
        return
    labels = top['Cluster_Label'].tolist() + (['Other'] if len(other) > 0 else [])
    values_by_group = {}
    for cat in categories:
        pcol = f'{cat}_Percent'
        vals = top[pcol].tolist()
        if len(other) > 0:
            vals.append(max(0.0, 100.0 - float(np.nansum(vals))))
        values_by_group[cat] = vals
    fig, ax = plt.subplots(figsize=(12, max(6, len(categories) * 0.4)))
    bottom = np.zeros(len(categories))
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    for i, lbl in enumerate(labels):
        heights = [values_by_group[cat][i] for cat in categories]
        ax.bar(categories, heights, bottom=bottom, label=lbl, color=colors[i], alpha=0.9)
        bottom += np.array(heights)
    fs_axis = _VIZ["chart"]["font_size_axis"]
    ax.set_ylabel('Percentage (%)', fontsize=fs_axis - 1, fontweight='bold', fontfamily=_VIZ["chart"]["font_family"])
    ax.set_title(_VIZ["titles"]["cluster_mix_stacked"], fontsize=fs_axis, fontweight='bold', pad=12, fontfamily=_VIZ["chart"]["font_family"])
    ax.set_ylim(0, 100)
    ax.legend(fontsize=_VIZ["chart"]["font_size_tick"] - 1, ncol=2, frameon=True, prop={"family": _VIZ["chart"]["font_family"]})
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_file, dpi=_VIZ["chart"]["dpi"] - 50 or 250, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {output_file}")


def create_cluster_facets(comparison_df, categories, output_file, max_groups=5, top_clusters_per_group=12):
    if not categories:
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
        bar_color = _VIZ["chart"]["bar_color"]
        ax.barh(top['Cluster_Label'], top[pct_col], color=bar_color, alpha=0.85)
        ax.set_title(str(cat), fontsize=_VIZ["chart"]["font_size_axis"], fontweight='bold', fontfamily=_VIZ["chart"]["font_family"])
        ax.set_xlabel('Percentage (%)', fontsize=_VIZ["chart"]["font_size_tick"] + 1, fontfamily=_VIZ["chart"]["font_family"])
        ax.grid(axis='x', alpha=0.25)
        ax.tick_params(axis='y', labelsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_axis_off()
    fig.suptitle(_VIZ["titles"]["cluster_facets"], fontsize=_VIZ["chart"]["font_size_title"], fontweight='bold', fontfamily=_VIZ["chart"]["font_family"])
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_file, dpi=_VIZ["chart"]["dpi"] - 50 or 250, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {output_file}")


def create_treemap_html(comparison_df, categories, output_file=None):
    output_file = output_file or _of["cluster_treemap"]
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
        fig = px.treemap(pd.DataFrame(df), path=[px.Constant("Clusters"), 'Cluster_Label'], values='Total_Count', color=color_col, color_continuous_scale='RdBu', hover_data=['Cluster_ID'] + percent_cols)
    else:
        if percent_cols:
            df['Dominant_Group'] = df[percent_cols].idxmax(axis=1).str.replace('_Percent', '', regex=False)
            df['Dominant_Share'] = df[percent_cols].max(axis=1)
        else:
            df['Dominant_Group'] = 'N/A'
            df['Dominant_Share'] = np.nan
        fig = px.treemap(pd.DataFrame(df), path=[px.Constant("Clusters"), 'Cluster_Label'], values='Total_Count', color='Dominant_Group', hover_data=['Cluster_ID', 'Dominant_Share'] + percent_cols)
    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10), title="Cluster treemap")
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"✓ Saved to {output_file}")
    return True


def create_sankey_html(comparison_df, categories, output_file=None, top_clusters=20):
    output_file = output_file or _of["cluster_sankey"]
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
    fig = go.Figure(data=[go.Sankey(node=dict(pad=15, thickness=15, line=dict(color="black", width=0.2), label=labels), link=dict(source=sources, target=targets, value=values))])
    fig.update_layout(title_text=f"Cluster → Group Sankey (top {top_clusters} clusters)", font_size=10)
    fig.write_html(output_file, include_plotlyjs='cdn')
    print(f"✓ Saved to {output_file}")
    return True


def _plotly_available():
    try:
        importlib.import_module("plotly.express")
        return True
    except Exception:
        return False


def fixed_outputs_mode():
    """Generate the fixed set of reports and visualizations."""
    if not _plotly_available():
        print("\nPlotly not available. Treemap and Sankey will be skipped.")
        print("Python used:", sys.executable)
        print("To enable: run in terminal:", sys.executable, "-m pip install plotly")
        print()
    comparison_df, metadata, clusters_df, metadata_with_clusters = load_fixed_mode_data()
    categories = get_categories_from_comparison(comparison_df)
    group_col = get_group_col_from_report()
    chi2, chi2_p = parse_chi_square_from_report()
    group_totals = _infer_group_totals_from_comparison(comparison_df, categories)
    if 'Cluster' in metadata_with_clusters.columns:
        total_rows = int((pd.to_numeric(metadata_with_clusters['Cluster'], errors='coerce').fillna(-1) >= 0).sum())
    else:
        total_rows = int(pd.to_numeric(clusters_df.get('Message_Count', 0), errors='coerce').fillna(0).sum())
    print("\n" + "=" * 60)
    print("CREATING REPORTS AND VISUALIZATIONS")
    print("=" * 60)
    print(f"Groups: {categories}")
    print(f"Group-by column (word clouds): {group_col or 'not found'}")
    generated = []

    def _run(name, fn, *out_names):
        try:
            fn()
            for n in out_names:
                if n is not None:
                    generated.append(n)
        except Exception as e:
            print(f"  [SKIP] {name}: {e}")

    _run("overview report", lambda: write_overview_report(clusters_df, total_rows, group_col, categories, group_totals, chi2, chi2_p), _of["overview_report"])
    _run("cluster sizes chart", lambda: create_cluster_sizes_chart(clusters_df, total_rows, _of["cluster_sizes"]), _of["cluster_sizes"])
    _run("cluster proportions pie", lambda: create_cluster_proportions_pie(clusters_df, total_rows, _of["cluster_proportions_pie"], top_n=10), _of["cluster_proportions_pie"])
    _run("cluster heatmap", lambda: create_cluster_heatmap(comparison_df, categories, _of["cluster_heatmap"]), _of["cluster_heatmap"])
    _run("cluster mix stacked bar", lambda: create_cluster_mix_stacked_bar(comparison_df, categories, _of["cluster_mix_stacked_bar"], top_clusters=10), _of["cluster_mix_stacked_bar"])
    if group_col and group_col in metadata.columns:
        for cat in categories:
            safe = re.sub(r'[^\w\-]', '_', str(cat))
            out = f'{_of["wordcloud_prefix"]}{safe}.png'
            _run(f"wordcloud {cat}", lambda o=out, c=cat: create_wordcloud(metadata, group_col, c, o), out)
    else:
        print("  Skipping word clouds (no group column in metadata)")
    _run("cluster facets", lambda: create_cluster_facets(comparison_df, categories, _of["cluster_facets"], max_groups=5, top_clusters_per_group=12), _of["cluster_facets"])
    try:
        if create_treemap_html(comparison_df, categories):
            generated.append(_of["cluster_treemap"])
    except Exception as e:
        print(f"  [SKIP] treemap: {e}")
    try:
        if create_sankey_html(comparison_df, categories, top_clusters=20):
            generated.append(_of["cluster_sankey"])
    except Exception as e:
        print(f"  [SKIP] sankey: {e}")
    _run("cluster comparison chart", lambda: create_cluster_comparison_chart(comparison_df, categories, _of["cluster_comparison_chart"]), _of["cluster_comparison_chart"] if len(categories) in (2, 3) else None)
    _run("difference chart", lambda: create_difference_chart(comparison_df, categories, _of["cluster_differences_chart"]), _of["cluster_differences_chart"] if ('Difference' in comparison_df.columns and 'P_Value' in comparison_df.columns) else None)

    print("\n" + "=" * 60)
    print("Step 3 Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    for f in generated:
        if f:
            print(f"  - {f}")
    print("=" * 60)


def llm_mode_interactive(confirm_before_run=False):
    """Interactive mode: user prompts for ad-hoc numbers or visualizations."""
    dfs = {}
    for name, path in [
        ("comparison_df", _of["comparison_csv"]),
        ("clusters_df", CLUSTERS_CSV),
        ("metadata_df", METADATA_CSV),
        ("metadata_with_clusters_df", METADATA_WITH_CLUSTERS_CSV),
    ]:
        if os.path.isfile(path):
            dfs[name] = pd.read_csv(path)
    if not dfs:
        raise FileNotFoundError("No input CSVs found. Run step 02 first and this step (comparison) once without --llm.")
    try:
        from dotenv import load_dotenv
        from google import genai
    except Exception as e:
        raise RuntimeError("LLM mode requires python-dotenv and google-genai. Install with: pip install python-dotenv google-genai") from e
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found. Put it in .env (same as step 02).")
    genai_client = genai.Client(api_key=api_key)
    try:
        px = importlib.import_module("plotly.express")
        go = importlib.import_module("plotly.graph_objects")
    except Exception:
        px = None
        go = None

    def build_context():
        parts = ["You are generating Python code only. No markdown fences. No explanations."]
        parts.append("Available pandas DataFrames (already loaded):")
        for df_name, df in dfs.items():
            parts.append(f"- {df_name}: {len(df):,} rows; columns={list(df.columns)}")
        parts.append("Rules: Use pandas/matplotlib/seaborn/plotly. Save matplotlib with plt.savefig('llm_output_1.png', dpi=200, bbox_inches='tight'). Save plotly with fig.write_html('llm_output_1.html', include_plotlyjs='cdn'). Print key results. Do not read/write other files.")
        return "\n".join(parts)

    model_names = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-2.5-pro']

    def strip_code_fences(text):
        t = text.strip()
        if t.startswith("```"):
            t = re.sub(r"^```[a-zA-Z]*\n", "", t)
            t = re.sub(r"\n```$", "", t)
        return t.strip()

    import builtins
    safe_builtins = {"print": print, "len": len, "range": range, "min": min, "max": max, "sum": sum, "abs": abs, "sorted": sorted, "list": list, "dict": dict, "set": set, "float": float, "int": int, "str": str, "enumerate": enumerate, "zip": zip, "__import__": builtins.__import__}
    exec_globals = {"__builtins__": safe_builtins, "pd": pd, "np": np, "plt": plt, "sns": sns, "WordCloud": WordCloud, "os": os, "re": re}
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
        prompt = f"{build_context()}\n\nUser request:\n{user_req}\n\nReturn only executable Python code."
        code = None
        last_err = None
        for model_name in model_names:
            try:
                resp = genai_client.models.generate_content(model=model_name, contents=prompt)
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
        exec_globals["LLM_OUTPUT_INDEX"] = counter
        try:
            exec(code, exec_globals, {})
        except Exception as e:
            print(f"\nError while running generated code: {e}")
        counter += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare cluster proportions and create reports/visualizations.'
    )
    parser.add_argument('--group-by', type=str, default=None,
                        help='Column name to group by. Required if 0 or 2+ category columns detected.')
    parser.add_argument('--groups', nargs='*', default=None,
                        help='Group values to compare (space-separated). If omitted, all values in the column are used.')
    parser.add_argument('--compare-only', action='store_true',
                        help='Only run comparison and write CSV + report; skip all visualizations.')
    parser.add_argument('--llm', action='store_true',
                        help='After comparison, run interactive LLM mode for ad-hoc numbers/visualizations.')
    parser.add_argument('--confirm', action='store_true',
                        help='In LLM mode, ask for confirmation before running generated code.')
    args = parser.parse_args()

    clusters_file = CLUSTERS_CSV
    metadata_file = METADATA_CSV
    metadata_with_clusters_file = METADATA_WITH_CLUSTERS_CSV
    output_csv = _of["comparison_csv"]
    output_report = _of["comparison_report"]

    try:
        clusters_df, metadata = load_data(clusters_file, metadata_file)
        metadata_with_clusters = pd.read_csv(metadata_with_clusters_file)
        category_columns = detect_category_columns(metadata_with_clusters)
        if args.group_by:
            if args.group_by not in metadata_with_clusters.columns:
                print(f"Error: Column '{args.group_by}' not found in {metadata_with_clusters_file}.")
                print(f"Available columns: {list(metadata_with_clusters.columns)}")
                exit(1)
            group_col = args.group_by
            print(f"Using group-by column: {group_col} (from CLI)")
        elif len(category_columns) == 1:
            group_col = category_columns[0]
            print(f"Using group-by column: {group_col} (only one category column found)")
        elif len(category_columns) == 0:
            print("Error: No category-like column found (limited unique string values).")
            print("Use --group-by COLUMN to specify which column to compare.")
            exit(1)
        else:
            print(f"Multiple category-like columns found: {category_columns}")
            print("Please specify one with: --group-by COLUMN")
            exit(1)

        all_categories = sorted(metadata_with_clusters[group_col].dropna().astype(str).str.strip().unique().tolist())
        if args.groups:
            categories = []
            for v in args.groups:
                categories.extend(x.strip() for x in str(v).split(",") if x.strip())
            categories = list(dict.fromkeys(categories))
            missing = [c for c in categories if c not in all_categories]
            if missing:
                print(f"Error: --groups values not found in column '{group_col}': {missing}")
                print(f"Available values: {all_categories}")
                exit(1)
            if len(categories) < 2:
                print("Error: --groups must specify at least 2 values to compare.")
                exit(1)
            print(f"Comparing selected groups: {categories}")
        else:
            categories = all_categories
        if len(categories) < 2:
            print(f"Error: Column '{group_col}' has only one value: {all_categories}. Need at least 2 groups.")
            exit(1)

        metadata_with_clusters = metadata_with_clusters[
            metadata_with_clusters[group_col].astype(str).str.strip().isin(categories)
        ].copy()

        comparison_df, totals = calculate_proportions(
            metadata_with_clusters, clusters_df, group_col, categories
        )
        chi2, chi2_p, dof, residuals_df = perform_chi_square_test(
            metadata_with_clusters, clusters_df, group_col
        )
        comparison_df = add_statistical_tests(comparison_df, totals, categories)

        sort_col = 'Difference' if len(categories) == 2 and 'Difference' in comparison_df.columns else f'{categories[0]}_Percent'
        comparison_df = comparison_df.sort_values(sort_col, key=abs if sort_col == 'Difference' else None, ascending=False)

        comparison_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n✓ Saved comparison results to {output_csv}")

        report = generate_report(comparison_df, chi2, chi2_p, totals, categories, group_col, residuals_df=residuals_df)
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Saved comparison report to {output_report}")

        print("\n" + report)

        if args.compare_only:
            print("\n" + "="*60)
            print("Comparison complete (--compare-only). No visualizations generated.")
            print("="*60)
            exit(0)
        if args.llm:
            llm_mode_interactive(confirm_before_run=args.confirm)
            exit(0)
        fixed_outputs_mode()

    except FileNotFoundError as e:
        print(f"Error: {e} not found. Please run previous steps first.")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
