#!/usr/bin/env python3
"""
Step 4: Compare clusters across a grouping column
Performs chi-square test and z-tests for proportions (when 2 groups) with confidence intervals.
Group-by column: auto-detected if exactly one category-like column; otherwise use --group-by.
"""

import argparse
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency
import os

# Columns to exclude from category detection
EXCLUDE_FROM_CATEGORY = {'Message', 'Cluster', 'Cluster_ID', 'Cluster_Label'}
MAX_CARDINALITY_FOR_CATEGORY = 50


def detect_category_columns(metadata_with_clusters):
    """
    Find columns that look like categories: string-like with limited unique values.
    Returns list of column names.
    """
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


def load_data(clusters_file, metadata_file):
    """Load clustering results and metadata."""
    print("Loading data...")
    clusters_df = pd.read_csv(clusters_file)
    metadata = pd.read_csv(metadata_file)
    print(f"✓ Loaded {len(clusters_df)} clusters")
    print(f"✓ Loaded {len(metadata):,} messages")
    return clusters_df, metadata


def calculate_proportions(metadata_with_clusters, clusters_df, group_col, categories):
    """Calculate cluster proportions for each group."""
    totals = {cat: len(metadata_with_clusters[metadata_with_clusters[group_col].astype(str) == str(cat)]) for cat in categories}
    total_all = sum(totals.values())
    print(f"\nGroup column: '{group_col}'")
    for cat in categories:
        print(f"  Total messages - {cat}: {totals[cat]:,}")
    if total_all == 0:
        raise ValueError(f"No messages in any of the groups: {categories}")

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
    return comparison_df, totals


def perform_z_test(n1, x1, n2, x2, alpha=0.05):
    """
    Z-test for difference in proportions (two groups).
    Returns: z_score, p_value, (ci_lower, ci_upper) in percentage points.
    """
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
    """Chi-square test: clusters x groups association."""
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
    return chi2, p_value, dof


def add_statistical_tests(comparison_df, totals, categories):
    """Add z-tests and CIs when there are exactly 2 groups."""
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


def generate_report(comparison_df, chi2, chi2_p, totals, categories, group_col):
    """Generate text report; handles 2 or more groups."""
    report_lines = []
    report_lines.append("="*60)
    report_lines.append(f"CLUSTER COMPARISON BY '{group_col}'")
    report_lines.append("="*60)
    report_lines.append(f"\nTotal messages per group:")
    for cat in categories:
        report_lines.append(f"  {cat}: {totals[cat]:,}")
    report_lines.append(f"\nChi-Square Test Results:")
    report_lines.append(f"  Chi-square statistic: {chi2:.4f}")
    report_lines.append(f"  P-value: {chi2_p:.6f}")
    report_lines.append(f"  Conclusion: {'Significant difference' if chi2_p < 0.05 else 'No significant difference'}")

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

    report_lines.append("\n" + "="*60)
    return "\n".join(report_lines)


if __name__ == '__main__':
    # CLI usage:
    #   python 04_compare_clusters.py
    #   python 04_compare_clusters.py --group-by COLUMN
    #   python 04_compare_clusters.py --group-by COLUMN --groups v1 v2 v3
    parser = argparse.ArgumentParser(
        description='Compare cluster proportions across a grouping column.'
    )
    parser.add_argument('--group-by', type=str, default=None,
                        help='Column name to group by. Required if 0 or 2+ category columns detected.')
    parser.add_argument('--groups', nargs='*', default=None,
                        help='Group values to compare (space-separated). If omitted, all values in the column are used.')
    args = parser.parse_args()

    clusters_file = 'clusters_with_labels.csv'
    metadata_file = 'embeddings_metadata.csv'
    metadata_with_clusters_file = 'metadata_with_clusters.csv'
    output_csv = 'cluster_comparison.csv'
    output_report = 'cluster_comparison_report.txt'

    try:
        clusters_df, metadata = load_data(clusters_file, metadata_file)
        metadata_with_clusters = pd.read_csv(metadata_with_clusters_file)

        # Resolve group-by column
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
        chi2, chi2_p, dof = perform_chi_square_test(
            metadata_with_clusters, clusters_df, group_col
        )
        comparison_df = add_statistical_tests(comparison_df, totals, categories)

        sort_col = 'Difference' if len(categories) == 2 and 'Difference' in comparison_df.columns else f'{categories[0]}_Percent'
        comparison_df = comparison_df.sort_values(sort_col, key=abs if sort_col == 'Difference' else None, ascending=False)

        comparison_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        print(f"\n✓ Saved comparison results to {output_csv}")

        report = generate_report(comparison_df, chi2, chi2_p, totals, categories, group_col)
        with open(output_report, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Saved comparison report to {output_report}")

        print("\n" + report)
        print("\n" + "="*60)
        print("Step 4 Complete! Cluster comparison by group done.")
        print("="*60)

    except FileNotFoundError as e:
        print(f"Error: {e} not found. Please run previous steps first.")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
