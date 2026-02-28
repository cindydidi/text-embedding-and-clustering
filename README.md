# Topic Analysis from Your Data — User Guide

This folder contains three scripts that analyze text from a CSV: they turn text rows into “topics,” cluster them, then compare topics by group and create reports and charts. **Run the scripts in order (01 → 02 → 03).** Each step uses the output from the previous one.

---

## What You Need Before You Start

### 1. Your data file (Step 1 only)

- **Any CSV file** in the same folder as the scripts. The script **auto-detects** your CSV: if there is **only one** `.csv` file in the folder, it uses that. If there are **multiple** CSV files, the script will ask you to specify which one with **`--input filename.csv`**.
- **Format:** Any CSV (e.g. Excel saved as “CSV”). The script needs **one column that contains the text** you want to analyze.

**Which column is used for text?**

- If your CSV has **only one column**, that column is used automatically.
- If your CSV has **exactly one column that looks like text** (not all numbers), that column is used automatically.
- If your CSV has **more than one text-like column**, the script will not guess. It will print an error and ask you to specify the column, for example:
  ```bash
  python 01_generate_embeddings.py --text-column "YourColumnHeader"
  ```
  Use the **exact column header** from your CSV (whatever that column is called).

**Other columns**

- **All columns** in your CSV are passed through to the output metadata. Rows where the chosen text column is empty are skipped.
- Step 3 compares clusters across **any category column** (for example `Channel` with values like `Website` / `Mobile`). It auto-detects a category-like column if there is exactly one; otherwise you specify it with `--group-by`.
- Use UTF-8 encoding so special characters (e.g. accents, Chinese) work correctly.

### 2. Python and packages

- **Python 3** installed on your computer.
- Required packages (install once):

```bash
pip install google-genai numpy pandas python-dotenv scikit-learn matplotlib scipy seaborn wordcloud hdbscan plotly
```

### 3. Google API key (for Steps 1 and 2)

- **Step 1** and **Step 2** call Google’s APIs (embeddings and topic labels).
- Create a file named **`.env`** in the same folder as the scripts.
- Put this line in `.env` (replace `YOUR_KEY_HERE` with your real key):

```
GOOGLE_API_KEY=YOUR_KEY_HERE
```

- Keep `.env` private; do not share it or upload it.

---

## Step-by-step: What to Run and What You Get

### Step 1 — Generate embeddings  
**Script:** `01_generate_embeddings.py`

**What it does:**  
Detects the CSV in the folder (or uses the one you specify), sends each text row to Google’s embedding API, and saves numeric “embeddings” that the next steps—or your own tools—can use.

**What you provide:**

- One CSV file in the **same folder** as the script. The script auto-detects it when there’s only one CSV. One column must contain the text to embed (see “Which column is used for text?” above).

**How to run:**

1. Open Terminal (or Command Prompt).
2. Go to the folder that contains the scripts and your CSV:
   ```bash
   cd path/to/embedding
   ```
3. Run (no arguments needed if there is only one CSV in the folder):
   ```bash
   python 01_generate_embeddings.py
   ```
   If the folder has **multiple CSV files**, the script will list them and ask you to specify which one:
   ```bash
   python 01_generate_embeddings.py --input your_filename.csv
   ```
   If your CSV has **multiple text columns**, the script will ask you to specify which column to use:
   ```bash
   python 01_generate_embeddings.py --text-column "YourColumnHeader"
   ```
   (Use the exact column header from your CSV.)

**Optional arguments (only when needed):**

- **Multiple CSVs:** specify which file:  
  `python 01_generate_embeddings.py --input filename.csv`
- **Multiple text columns:** specify which column:  
  `python 01_generate_embeddings.py --text-column "YourColumnHeader"`
- Test with fewer rows:  
  `python 01_generate_embeddings.py --test 1000`
- Skip the first N rows:  
  `python 01_generate_embeddings.py --skip 5000`
- Use fewer parallel workers (if you get rate limits):  
  `python 01_generate_embeddings.py --workers 3`

**What you get:**

- **`embeddings.npy`** — numeric embeddings (used by Steps 2 and 3, or any other tool).
- **`embeddings_metadata.csv`** — one row per text row; the embedded text is in a column named **text**, and all other columns from your CSV are included.

**Rough time:** Depends on how many rows you have; the script prints progress and estimated time.

---

### Step 2 — Group text rows into topics and name them  
**Script:** `02_cluster_and_extract.py`

**What it does:**  
Groups text rows into clusters (topics) using the embeddings, then gives each cluster a short topic name (using Google’s API when possible, or simple keywords as fallback). **Default is HDBSCAN**, which finds the number of clusters automatically (no K needed). If you use K-means or Agglomerative, the number of clusters (K) comes from `--k`, from an existing `optimal_k.txt`, or from automatic detection (Elbow + Silhouette); when detected, it writes `optimal_k.txt` and `optimal_clusters.png`.

**What you provide:**

- Outputs from Step 1: `embeddings.npy`, `embeddings_metadata.csv`. Optional: `optimal_k.txt` (for K-means or Agglomerative if you already have it).

**How to run:**

```bash
python 02_cluster_and_extract.py
```

**Optional arguments:**

- Use a fixed number of clusters (for kmeans/agglomerative only):  
  `python 02_cluster_and_extract.py --k 15`
- Use keyword-only labels (no Google API):  
  `python 02_cluster_and_extract.py --no-llm`
- Use K-means (K from --k, optimal_k.txt, or auto-detection):  
  `python 02_cluster_and_extract.py --algorithm kmeans`
- Use Agglomerative clustering:  
  `python 02_cluster_and_extract.py --algorithm agglomerative`

**What you get:**

- **`clusters_with_labels.csv`** — one row per cluster: cluster label, cluster ID, row count, sample text, and (if your data has a Channel column) counts by channel.
- **`metadata_with_clusters.csv`** — your metadata from Step 1 plus a **Cluster** column showing which topic each row belongs to.
- When K is auto-detected (kmeans/agglomerative without `--k` or `optimal_k.txt`): **`optimal_k.txt`**, **`optimal_clusters.png`**.

---

### Step 3 — Compare clusters and create reports/visualizations  
**Script:** `03_analyze_and_visualize.py`

**What it does:**  
Compares how often each cluster appears across values of a category column (e.g. `Channel`): runs an overall chi-square test; for **2 groups** also runs per-cluster two-proportion z-tests and reports significant differences. Then it produces a fixed set of reports and charts (overview, cluster sizes, pie, heatmap, stacked bar, word clouds, comparison bar, difference chart, facets, treemap, Sankey). You can run **comparison only** with `--compare-only` (no charts), or **interactive LLM mode** with `--llm` for ad-hoc numbers/visuals (skips the fixed outputs in that mode).

**What you provide:**

- Step 2 outputs: `clusters_with_labels.csv`, `metadata_with_clusters.csv`, and (for word clouds) `embeddings_metadata.csv`. At least one category-like column, or specify one with `--group-by`.

**How to run:**

```bash
python 03_analyze_and_visualize.py
```

**Optional arguments:**

- **Comparison only (no charts):**  
  `python 03_analyze_and_visualize.py --compare-only`
- **Interactive LLM mode (ad-hoc numbers/visuals):**  
  `python 03_analyze_and_visualize.py --llm`  
  With confirmation before running generated code:  
  `python 03_analyze_and_visualize.py --llm --confirm`
- **Group-by column:**  
  `python 03_analyze_and_visualize.py --group-by COLUMN`
- **Specific groups:**  
  `python 03_analyze_and_visualize.py --group-by COLUMN --groups v1 v2 v3`

**What you get:**

**Comparison (always):**

- **`cluster_comparison.csv`** — each cluster with counts and percentages per group, and (for 2 groups) difference and p-values.
- **`cluster_comparison_report.txt`** — summary: total rows per group, chi-square result, top clusters per group, and (for 2 groups) which clusters differ most.

**Reports and charts (unless you use `--compare-only` or `--llm`):**

- **`overview_report.txt`** — overview text summary (total messages, number of clusters, group column + group counts, cluster sizes).
- **`cluster_sizes.png`** — horizontal bar chart of cluster sizes.
- **`cluster_proportions_pie.png`** — pie chart of cluster proportions (top clusters + Other).
- **`cluster_heatmap.png`** — heatmap of cluster percentages by group.
- **`cluster_mix_stacked_bar.png`** — 100% stacked bar chart: cluster mix within each group (top clusters + Other).
- **`cluster_comparison_chart.png`** — bar chart: top clusters by group percentages (2 or 3 groups only).
- **`cluster_differences_chart.png`** — significant cluster differences (2 groups only).
- **`cluster_distribution_facets.png`** — small multiples: one panel per group (top 5 groups when many).
- **`cluster_treemap.html`** — interactive treemap of clusters (size by count; color by dominant group or 2-group difference).
- **`cluster_group_sankey.html`** — interactive Sankey: clusters → groups (top clusters).
- **`wordcloud_<Group>.png`** — one word cloud per group (if the group-by column is in the metadata).

---


## Quick reference: input and output by step

| Step | Script                       | Main input                    | Main output |
|------|------------------------------|-------------------------------|-------------|
| 1    | `01_generate_embeddings.py`   | Your CSV (auto-detected, or `--input` if multiple) | `embeddings.npy`, `embeddings_metadata.csv` |
| 2    | `02_cluster_and_extract.py`  | `embeddings.npy`, `embeddings_metadata.csv` (optional: `optimal_k.txt` for kmeans/agglomerative) | `clusters_with_labels.csv`, `metadata_with_clusters.csv` (+ optional `optimal_k.txt`, `optimal_clusters.png`) |
| 3    | `03_analyze_and_visualize.py`| Step 2 outputs                | `cluster_comparison.csv`, `cluster_comparison_report.txt`, then (unless `--compare-only` or `--llm`) overview report + all charts/HTML (see Step 3 section) |

---

## If something goes wrong

- **“No CSV file found” / “Multiple CSV files found”**  
  Put exactly one CSV in the same folder as the script so it can auto-detect it, or use **`--input filename.csv`** to specify which file when there are multiple.

- **“Please specify which column to use for embeddings”**  
  Your CSV has more than one text-like column. Run with **`--text-column "YourColumnHeader"`** (the exact header of the column you want to embed).

- **“File not found” (Steps 2–3)**  
  Run the steps in order (01 → 02 → 03) and keep all generated files in the same folder. If you use **`--test`** in Step 1, it creates `embeddings_test.npy` and `embeddings_metadata_test.csv`; later steps by default look for `embeddings.npy` and `embeddings_metadata.csv`.

- **“GOOGLE_API_KEY not found”**  
  Create a **`.env`** file in the script folder with:  
  `GOOGLE_API_KEY=your_actual_key`

- **Import or package errors**  
  Install the required packages (see “Python and packages” above).

- **Group comparison is empty**  
  Step 3 compares clusters by a category column (often `Channel`). If your CSV doesn’t have a suitable category column (or you don’t specify one with `--group-by`), group comparison outputs and some charts will be empty.

---

## Summary

1. Put **your CSV** in the same folder as the scripts (one CSV so it auto-detects, or use **`--input filename.csv`** when there are multiple). One column must contain the text to analyze; use **`--text-column "YourColumnHeader"`** if the CSV has multiple text columns.  
2. Add **`.env`** with **GOOGLE_API_KEY** for Steps 1 and 2.  
3. Install Python and the required packages.  
4. Run in order: **01 → 02 → 03**.  
5. Use the generated CSVs for detailed analysis and the PNGs/HTML for presentations or reports.
