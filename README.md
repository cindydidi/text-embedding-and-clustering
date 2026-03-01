# Text Embedding and Clustering

**heads-up:** This project was vibe coded. The author doesn’t know a single line of code. Please bear with any mistakes, redundancy, or places where the style isn’t exactly graceful.

---

## What this does

Use this toolkit to **embed** CSV text into numerical vectors, **cluster** related data points, and **visualize** the results through interactive reports and charts.

Choose your workflow: Use the **Web App** for an intuitive, no-code experience, or run the **CLI scripts** for manual control and automation.

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/cindydidi/text-embedding-and-clustering
cd text-embedding-and-clustering
pip install -r requirements.txt
```
---

## Web app

### Setup

Run the pipeline in a **web interface**

1. **Python 3** on your computer.
2. After cloning and installing (see [Installation](#installation) above), create a **`.env`** file in this folder (same folder as `web_app.py`) with your Google API key:
   ```
   GOOGLE_API_KEY=your_key_here
   ```
   Keep `.env` private and don’t share it.

### Run the app

```bash
streamlit run web_app.py
```

Open the URL that appears (usually **http://localhost:8501**). Upload a CSV, pick the text column if asked, then use the three buttons to run Embedding, Clustering, and Data/Visualization. Results and downloads appear at the bottom. Config (sentiment words, chart colors, etc.) can be edited in the sidebar.

---

## Running the scripts from the command line

You can also run the pipeline as three separate steps:

1. **Step 1** — `01_generate_embeddings.py` — reads your CSV and creates embeddings.
2. **Step 2** — `02_cluster_and_extract.py` — clusters the embeddings and adds topic labels.
3. **Step 3** — `03_analyze_and_visualize.py` — compares clusters and generates reports and charts.

Run them in that order. Put your CSV in the same folder (or point to it with `--input`). For all options, run any script with `--help`, for example:

```bash
python 01_generate_embeddings.py --help
```

---

## If something goes wrong

- **“GOOGLE_API_KEY not found”** — Add a `.env` file in this folder with `GOOGLE_API_KEY=your_key`.
- **Missing packages** — Run `pip install -r requirements.txt`.
- **File not found (steps 2 or 3)** — Run the steps in order (01 → 02 → 03) and keep the generated files in the same folder (or use the web app, which uses a `streamlit_workspace` folder).
- **Multiple text columns** — In the web app, choose the column from the dropdown. From the command line, use `--text-column "YourColumnName"` with step 1.

---

## Summary

- **Web app:** `pip install -r requirements.txt`, add `.env` with `GOOGLE_API_KEY`, then `streamlit run web_app.py`. Use the UI to upload CSV and run the pipeline.
- **Command line:** Run `01_generate_embeddings.py` → `02_cluster_and_extract.py` → `03_analyze_and_visualize.py` in order; use `--help` on any script for options.
