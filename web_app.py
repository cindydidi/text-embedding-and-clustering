#!/usr/bin/env python3
"""
Streamlit web app for the embedding pipeline: upload CSV, run embedding → clustering → visualization,
edit config, and view results. Modern UI with Noto Sans and gradient accents.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

# Load .env from repo root so subprocesses inherit GOOGLE_API_KEY (they run with cwd=workspace)
REPO_ROOT = Path(__file__).resolve().parent
try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv(REPO_ROOT / ".env")
except Exception:
    pass

import pandas as pd
import streamlit as st

WORKSPACE = REPO_ROOT / "streamlit_workspace"
WORKSPACE.mkdir(exist_ok=True)

CONFIG_PATH = REPO_ROOT / "config" / "config.json"
SCRIPT_01 = REPO_ROOT / "01_generate_embeddings.py"
SCRIPT_02 = REPO_ROOT / "02_cluster_and_extract.py"
SCRIPT_03 = REPO_ROOT / "03_analyze_and_visualize.py"

INPUT_CSV = "input.csv"
EMB_NPY = "embeddings.npy"
EMB_META = "embeddings_metadata.csv"
CLUSTERS_CSV = "clusters_with_labels.csv"
META_CLUSTERS_CSV = "metadata_with_clusters.csv"


def inject_css():
    """Noto Sans + modern layout + gradient accents."""
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    /* Global font and modern base */
    html, body, [class*="css"] {
        font-family: 'Noto Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    /* Gradient header */
    .gradient-header {
        background: linear-gradient(135deg, #0f766e 0%, #0e7490 50%, #6366f1 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 4px 14px rgba(15, 118, 110, 0.25);
    }
    .gradient-header h1 { margin: 0; font-size: 1.6rem; font-weight: 700; }
    .gradient-header p { margin: 0.4rem 0 0 0; opacity: 0.95; font-size: 0.95rem; }
    /* Sidebar gradient */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0fdfa 0%, #ecfeff 50%, #eef2ff 100%) !important;
    }
    /* Primary button gradient (via container) */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: box-shadow 0.2s;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(15, 118, 110, 0.3);
    }
    /* Cards / expanders */
    .stExpander {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    </style>
    """, unsafe_allow_html=True)


def load_config():
    if not CONFIG_PATH.exists():
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def run_script(script_path, args, cwd):
    """Run a Python script with env from parent (so GOOGLE_API_KEY is set)."""
    cmd = [sys.executable, str(script_path)] + args
    env = {**os.environ}
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    return r.returncode, r.stdout, r.stderr


def workspace_has(*names):
    return all((WORKSPACE / n).exists() for n in names)


def get_string_columns(df, sample_size=500):
    """Heuristic: columns that look like text (mostly non-numeric)."""
    if df is None or df.empty:
        return []
    cols = []
    for c in df.columns:
        s = df[c].dropna().astype(str)
        if len(s) == 0:
            continue
        sample = s.head(sample_size)
        non_numeric = sum(1 for v in sample if not v.replace(".", "").replace("-", "").strip().isdigit())
        if non_numeric / max(len(sample), 1) > 0.5:
            cols.append(c)
    return cols


# --- Page config and CSS ---
st.set_page_config(page_title="Embedding Pipeline", page_icon="📊", layout="wide")
inject_css()

# --- Header ---
st.markdown("""
<div class="gradient-header">
<h1>📊 Embedding and Clustering Pipeline</h1>
<p>Upload your data in CSV, run embedding → clustering → visualization in one single pipeline. Edit config in the sidebar.</p>
</div>
""", unsafe_allow_html=True)

# --- Session state ---
if "last_stdout" not in st.session_state:
    st.session_state.last_stdout = {}
if "last_stderr" not in st.session_state:
    st.session_state.last_stderr = {}

# --- Sidebar: Config editor ---
with st.sidebar:
    st.subheader("⚙️ Config")
    config = load_config()

    with st.expander("Sentiment words", expanded=True):
        neg = config.get("sentiment_words", {}).get("negative", [])
        pos = config.get("sentiment_words", {}).get("positive", [])
        neg_text = st.text_area("Negative (comma or newline)", value="\n".join(neg) if isinstance(neg, list) else str(neg), height=120)
        pos_text = st.text_area("Positive (comma or newline)", value="\n".join(pos) if isinstance(pos, list) else str(pos), height=120)

    with st.expander("Chart & colors", expanded=True):
        viz = config.get("visualization", {})
        chart = viz.get("chart", {})
        bar_color = st.text_input("Bar color (HEX or RGB)", value=chart.get("bar_color", "#2E86AB"), key="bar_color")
        bar_color_picker = st.color_picker("Bar color picker", value=chart.get("bar_color", "#2E86AB"), key="bar_color_pick")
        if bar_color_picker:
            bar_color = bar_color_picker
        diff_pos = st.text_input("Difference positive (HEX)", value=chart.get("difference_positive_color", "#2E86AB"), key="diff_pos")
        diff_neg = st.text_input("Difference negative (HEX)", value=chart.get("difference_negative_color", "#A23B72"), key="diff_neg")
        wc_bg = st.text_input("Wordcloud background (HEX)", value=chart.get("wordcloud_background", "white"), key="wc_bg")
        font_family = st.text_input("Chart font family", value=chart.get("font_family", "DejaVu Sans"), key="font_fam")
        dpi = st.number_input("DPI", min_value=72, max_value=600, value=int(chart.get("dpi", 300)), key="dpi")

    with st.expander("Advanced: edit raw JSON", expanded=False):
        raw_json = st.text_area("JSON", value=json.dumps(config, indent=2), height=200, key="raw_json")
        if st.button("Save raw JSON", key="save_raw"):
            try:
                parsed = json.loads(raw_json)
                save_config(parsed)
                st.success("Saved.")
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")

    if st.button("💾 Save config (structured)", key="save_structured"):
        def parse_words(t):
            out = []
            for line in t.replace(",", "\n").splitlines():
                w = line.strip()
                if w:
                    out.append(w)
            return out

        if "sentiment_words" not in config:
            config["sentiment_words"] = {}
        config["sentiment_words"]["negative"] = parse_words(neg_text)
        config["sentiment_words"]["positive"] = parse_words(pos_text)
        if "visualization" not in config:
            config["visualization"] = {}
        if "chart" not in config["visualization"]:
            config["visualization"]["chart"] = {}
        config["visualization"]["chart"]["bar_color"] = bar_color
        config["visualization"]["chart"]["difference_positive_color"] = diff_pos
        config["visualization"]["chart"]["difference_negative_color"] = diff_neg
        config["visualization"]["chart"]["wordcloud_background"] = wc_bg
        config["visualization"]["chart"]["font_family"] = font_family
        config["visualization"]["chart"]["dpi"] = dpi
        save_config(config)
        st.success("Config saved.")

# --- Main: CSV upload for Step 1 ---
st.subheader("📁 Upload your CSV")
uploaded = st.file_uploader("Choose a CSV", type=["csv"])
text_column_override = None
if uploaded is not None:
    df_preview = pd.read_csv(uploaded, nrows=5)
    st.dataframe(df_preview, width="stretch")
    all_cols = list(df_preview.columns)
    string_cols = get_string_columns(pd.read_csv(uploaded, nrows=500))
    if not string_cols:
        string_cols = all_cols
    if len(string_cols) > 1:
        text_column_override = st.selectbox("Text column for embedding", options=string_cols, key="text_col_select")
    else:
        text_column_override = string_cols[0] if string_cols else all_cols[0]
    path = WORKSPACE / INPUT_CSV
    with open(path, "wb") as f:
        f.write(uploaded.getvalue())
    st.caption(f"Saved as `{INPUT_CSV}` in workspace.")
elif (WORKSPACE / INPUT_CSV).exists():
    # Already have CSV in workspace from before; offer column choice for re-runs
    try:
        df_ws = pd.read_csv(WORKSPACE / INPUT_CSV, nrows=500)
        all_cols = list(df_ws.columns)
        string_cols = get_string_columns(df_ws)
        if not string_cols:
            string_cols = all_cols
        if len(string_cols) >= 1:
            text_column_override = st.selectbox("Text column for embedding (from existing file)", options=string_cols, key="text_col_ws")
    except Exception:
        pass

# --- Optional uploads: Clustering inputs ---
with st.expander("Use your own embedding outputs (optional)"):
    emb_npy = st.file_uploader("embeddings.npy", type=["npy"], key="up_emb_npy")
    emb_meta = st.file_uploader("embeddings_metadata.csv", type=["csv"], key="up_emb_meta")
    if emb_npy and emb_meta:
        with open(WORKSPACE / EMB_NPY, "wb") as f:
            f.write(emb_npy.getvalue())
        emb_meta_df = pd.read_csv(emb_meta)
        emb_meta_df.to_csv(WORKSPACE / EMB_META, index=False, encoding="utf-8-sig")
        st.success("Saved to workspace. You can run Clustering.")

# --- Optional uploads: Visualization inputs ---
with st.expander("Use your own clustering outputs (optional)"):
    up_clusters = st.file_uploader("clusters_with_labels.csv", type=["csv"], key="up_clusters")
    up_meta_cl = st.file_uploader("metadata_with_clusters.csv", type=["csv"], key="up_meta_cl")
    if up_clusters and up_meta_cl:
        pd.read_csv(up_clusters).to_csv(WORKSPACE / CLUSTERS_CSV, index=False, encoding="utf-8-sig")
        pd.read_csv(up_meta_cl).to_csv(WORKSPACE / META_CLUSTERS_CSV, index=False, encoding="utf-8-sig")
        st.success("Saved to workspace. You can run Visualization.")
    up_emb_meta_viz = st.file_uploader("embeddings_metadata.csv (optional, for word clouds)", type=["csv"], key="up_emb_meta_viz")
    if up_emb_meta_viz:
        pd.read_csv(up_emb_meta_viz).to_csv(WORKSPACE / EMB_META, index=False, encoding="utf-8-sig")
        st.caption("Saved embeddings_metadata.csv for word clouds.")

# --- Pipeline buttons ---
st.subheader("▶ Run pipeline")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**1. Embedding**")
    can_embed = (WORKSPACE / INPUT_CSV).exists()
    if not can_embed:
        st.caption("Make sure your CSV is uploaded above.")
    embed_workers = st.number_input("Workers", min_value=1, max_value=12, value=6, key="ew")
    embed_test = st.number_input("Test rows (0 = all)", min_value=0, value=0, key="et")
    if st.button("Run Embedding", disabled=not can_embed, key="btn_embed"):
        args = ["--input", str(WORKSPACE / INPUT_CSV)]
        if text_column_override:
            args += ["--text-column", text_column_override]
        if embed_workers:
            args += ["--workers", str(embed_workers)]
        if embed_test and embed_test > 0:
            args += ["--test", str(embed_test)]
        with st.spinner("Running..."):
            code, out, err = run_script(SCRIPT_01, args, WORKSPACE)
        st.session_state.last_stdout["embed"] = out
        st.session_state.last_stderr["embed"] = err
        if code == 0:
            st.success("Done.")
        else:
            st.error("Failed.")
        with st.expander("Log"):
            st.text(out[-3000:] if len(out) > 3000 else out)
            if err:
                st.text(err[-2000:] if len(err) > 2000 else err)

with col2:
    st.markdown("**2. Clustering**")
    can_cluster = workspace_has(EMB_NPY, EMB_META)
    if not can_cluster:
        st.caption("Run Embedding or upload embeddings above.")
    use_llm = st.checkbox("Use LLM for cluster labels", value=True, key="use_llm")
    algorithm = st.selectbox("Algorithm", ["hdbscan", "kmeans", "agglomerative"], key="algo")
    assign_noise = st.checkbox("Assign noise to nearest cluster", value=False, key="assign_noise")
    use_sentiment = st.checkbox("Use sentiment-aided clustering", value=True, key="use_sentiment")
    k_val = st.number_input("K (kmeans/agglomerative)", min_value=2, value=10, key="k_val")
    if st.button("Run Clustering", disabled=not can_cluster, key="btn_cluster"):
        args = []
        if not use_llm:
            args.append("--no-llm")
        args += ["--algorithm", algorithm]
        if assign_noise:
            args.append("--assign-noise")
        if not use_sentiment:
            args.append("--no-sentiment")
        if algorithm in ("kmeans", "agglomerative"):
            args += ["--k", str(k_val)]
        with st.spinner("Running..."):
            code, out, err = run_script(SCRIPT_02, args, WORKSPACE)
        st.session_state.last_stdout["cluster"] = out
        st.session_state.last_stderr["cluster"] = err
        if code == 0:
            st.success("Done.")
        else:
            st.error("Failed.")
        with st.expander("Log"):
            st.text(out[-3000:] if len(out) > 3000 else out)
            if err:
                st.text(err[-2000:] if len(err) > 2000 else err)

with col3:
    st.markdown("**3. Data / Visualization**")
    can_viz = workspace_has(CLUSTERS_CSV, META_CLUSTERS_CSV)
    if not can_viz:
        st.caption("Run Clustering or upload cluster outputs above.")
    meta_path = WORKSPACE / META_CLUSTERS_CSV
    group_by_col = None
    if meta_path.exists():
        try:
            mdf = pd.read_csv(meta_path, nrows=1000)
            exclude = {"text", "Message", "Cluster", "Cluster_ID", "Cluster_Label"}
            cat_candidates = [c for c in mdf.columns if c not in exclude and mdf[c].nunique() <= 50]
            if cat_candidates:
                group_by_col = st.selectbox("Group by column", ["Auto"] + cat_candidates, key="group_by")
        except Exception:
            pass
    groups_text = st.text_input("Groups (optional, space-separated)", placeholder="e.g. Website Mobile", key="groups")
    use_llm_viz = st.checkbox("Interactive LLM mode", value=False, key="use_llm_viz")
    if st.button("Run Visualization", disabled=not can_viz, key="btn_viz"):
        args = []
        if group_by_col and group_by_col != "Auto":
            args += ["--group-by", group_by_col]
        if groups_text.strip():
            args += ["--groups"] + groups_text.strip().split()
        if use_llm_viz:
            args.append("--llm")
        with st.spinner("Running..."):
            code, out, err = run_script(SCRIPT_03, args, WORKSPACE)
        st.session_state.last_stdout["viz"] = out
        st.session_state.last_stderr["viz"] = err
        if code == 0:
            st.success("Done.")
        else:
            st.error("Failed.")
        with st.expander("Log"):
            st.text(out[-3000:] if len(out) > 3000 else out)
            if err:
                st.text(err[-2000:] if len(err) > 2000 else err)

# --- Results ---
st.subheader("📂 Results")
out_files = list(WORKSPACE.iterdir()) if WORKSPACE.exists() else []
out_files = [f for f in out_files if f.is_file()]

for f in sorted(out_files, key=lambda x: x.name):
    name = f.name
    if name.endswith(".csv"):
        try:
            df = pd.read_csv(f, nrows=100)
            st.markdown(f"**{name}**")
            st.dataframe(df, width="stretch")
            with open(f, "rb") as fp:
                st.download_button(f"Download {name}", data=fp.read(), file_name=name, mime="text/csv", key=f"dl_{name}")
        except Exception:
            with open(f, "rb") as fp:
                st.download_button(f"Download {name}", data=fp.read(), file_name=name, key=f"dl_{name}")
    elif name.endswith(".png"):
        st.markdown(f"**{name}**")
        st.image(str(f), width="stretch")
        with open(f, "rb") as fp:
            st.download_button(f"Download {name}", data=fp.read(), file_name=name, mime="image/png", key=f"dl_{name}")
    elif name.endswith(".html"):
        st.markdown(f"**{name}**")
        with open(f, "rb") as fp:
            st.download_button(f"Download {name}", data=fp.read(), file_name=name, mime="text/html", key=f"dl_{name}")
    elif name.endswith((".txt", ".npy")):
        with open(f, "rb") as fp:
            st.download_button(f"Download {name}", data=fp.read(), file_name=name, key=f"dl_{name}")
