#!/usr/bin/env python3
"""Streamlit app: upload CSV, run embedding → clustering → visualization; edit config; view results."""

import io
import json
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

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

STEP_1_FILES = [INPUT_CSV, EMB_NPY, EMB_META]
STEP_2_FILES = [CLUSTERS_CSV, META_CLUSTERS_CSV]


def _step_3_filenames():
    """Step 3 output filenames from config plus wordcloud_*.png."""
    names = []
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            of = cfg.get("visualization", {}).get("output_files", {})
            for k, v in of.items():
                if k == "wordcloud_prefix":
                    continue
                if isinstance(v, str):
                    names.append(v)
        except Exception:
            pass
    return names


def _files_for_step(step: int, all_names: set) -> list:
    """Sorted filenames for this step that exist in workspace."""
    if step == 1:
        candidates = STEP_1_FILES
    elif step == 2:
        candidates = STEP_2_FILES
    else:
        candidates = list(_step_3_filenames())
        candidates += [n for n in all_names if n.startswith("wordcloud_") and n.endswith(".png")]
    return sorted(n for n in candidates if n in all_names)


def _make_zip_bytes(paths: list) -> bytes:
    """Zip given file paths into bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            if p.is_file():
                zf.write(p, p.name)
    buf.seek(0)
    return buf.getvalue()


def inject_css():
    """Inject Noto Sans, gradient header, frosted panels."""
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
    /* Page background: soft gradient instead of white */
    .stApp, [data-testid="stAppViewContainer"] {
        background: linear-gradient(165deg, #f0fdfa 0%, #ecfeff 25%, #fefce8 50%, #fef9c3 100%) !important;
    }
    /* Top bar: full-width transparent background */
    header[data-testid="stHeader"],
    .stApp header {
        background: transparent !important;
        width: 100% !important;
        left: 0 !important;
        right: 0 !important;
    }
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Noto Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    /* Pill-shaped title banner: fit to text, compact */
    .gradient-header {
        display: inline-block;
        background: linear-gradient(135deg, #86efac 0%, #bef264 50%, #fde047 100%);
        color: #14532d;
        padding: 0.5rem 1.25rem;
        border-radius: 9999px;
        margin-bottom: 1.5rem;
        border: none;
        box-shadow: none;
        backdrop-filter: blur(12px);
    }
    .gradient-header h1 { margin: 0; font-size: 1.35rem; font-weight: 700; }
    .gradient-header p {
        margin: 0.4rem 0 0.5rem 0;
        padding: 0;
        text-align: center;
        opacity: 0.9;
        font-size: 0.85rem;
        color: #166534;
    }
    /* Action blocks: each column gets a glass bar/block, no border */
    section[data-testid="stVerticalBlock"] > div[data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"] {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: none;
    }
    /* Sidebar: very light blue/green tint, frosted */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(240, 253, 250, 0.95) 0%, rgba(236, 254, 255, 0.95) 50%, rgba(254, 249, 195, 0.6) 100%) !important;
        backdrop-filter: blur(12px);
    }
    [data-testid="stSidebar"] > div { background: transparent !important; }
    /* Expanders / blocks: glass panels, no border */
    .stExpander {
        border-radius: 12px;
        border: none;
        background: rgba(255, 255, 255, 0.45);
        backdrop-filter: blur(8px);
        box-shadow: none;
    }
    /* Main blocks: no border, blend with background */
    [data-testid="stVerticalBlock"] > div {
        border-radius: 12px;
        border: none;
    }
    [data-testid="stExpander"] {
        border: none !important;
    }
    /* Primary button */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: box-shadow 0.2s;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(34, 197, 94, 0.25);
    }
    /* Embedding CSV upload: button-style, hide default text and Browse, show limit on info icon */
    .csv-upload-info { color: #9ca3af; cursor: help; font-size: 1rem; display: inline-block; vertical-align: middle; }
    main [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:first-of-type [data-testid="stFileUploader"]:first-of-type section {
        min-height: 2rem !important;
        padding: 0.35rem 0.75rem !important;
        border-radius: 8px !important;
        border: 1px solid #e5e7eb !important;
        background: #ffffff !important;
    }
    main [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:first-of-type [data-testid="stFileUploader"]:first-of-type section div[data-testid="stFileUploaderDropzoneInstructions"],
    main [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:first-of-type [data-testid="stFileUploader"]:first-of-type p,
    main [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:first-of-type [data-testid="stFileUploader"]:first-of-type small,
    main [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:first-of-type [data-testid="stFileUploader"]:first-of-type a {
        display: none !important;
    }
    main [data-testid="stHorizontalBlock"]:first-of-type [data-testid="column"]:first-of-type [data-testid="stFileUploader"]:first-of-type section::before {
        content: "Upload your CSV";
        font-weight: 500;
        color: #374151;
    }
    /* Hide header Deploy button and main menu (Wide mode) */
    .stAppDeployButton { visibility: hidden !important; }
    #MainMenu { visibility: hidden !important; }
    /* Prevent sidebar from being collapsed by user */
    [data-testid="collapsedControl"] { display: none !important; }
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


# Regex to parse embedding script progress: "Progress: 45.2% (4,521/10,000) | Rate: 12.3 text/s | ETA: 7.4 min"
_PROGRESS_RE = re.compile(
    r"Progress:\s*([\d.]+)%\s*\([\d,]+\/[\d,]+\)\s*\|\s*Rate:\s*([\d.]+)\s*text/s\s*\|\s*ETA:\s*([\d.]+)\s*min"
)


def _read_stdout_parse_progress(process, latest):
    """Read stdout; parse progress into latest[0] = (pct, rate, eta)."""
    buf = ""
    while True:
        chunk = process.stdout.read(4096)
        if not chunk:
            break
        buf += chunk
        for part in buf.split("\r"):
            m = _PROGRESS_RE.search(part)
            if m:
                latest[0] = (float(m.group(1)) / 100.0, float(m.group(2)), float(m.group(3)))
        if "\r" in buf:
            buf = buf[buf.rfind("\r") + 1 :]


def run_embedding_with_progress(script_path, args, cwd):
    """Run embedding script with Popen; stream stdout, parse progress; show progress bar + ETA. Returns (returncode, stdout, stderr)."""
    cmd = [sys.executable, str(script_path)] + args
    env = {**os.environ}
    process = subprocess.Popen(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env, bufsize=1
    )
    latest = [None]
    reader = threading.Thread(target=_read_stdout_parse_progress, args=(process, latest))
    reader.daemon = True
    reader.start()

    progress_placeholder = st.empty()
    last_pct, last_rate, last_eta = 0.0, 0.0, 0.0
    while process.poll() is None:
        if latest[0] is not None:
            last_pct, last_rate, last_eta = latest[0]
        with progress_placeholder.container():
            st.progress(min(1.0, last_pct))
            if last_rate > 0 or last_eta > 0:
                st.caption(f"Rate: {last_rate:.1f} text/s · ETA: {last_eta:.1f} min")
            else:
                st.caption("Starting…")
        time.sleep(0.3)
    reader.join(timeout=1.0)
    stdout, stderr = process.communicate()
    progress_placeholder.empty()
    return process.returncode, stdout or "", stderr or ""


def run_script(script_path, args, cwd):
    """Run a Python script with env from parent (so GOOGLE_API_KEY is set)."""
    cmd = [sys.executable, str(script_path)] + args
    env = {**os.environ}
    r = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, env=env)
    return r.returncode, r.stdout, r.stderr


def workspace_has(*names):
    return all((WORKSPACE / n).exists() for n in names)


def get_string_columns(df, sample_size=500):
    """Columns that look like text (mostly non-numeric)."""
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
st.set_page_config(page_title="Embedding and Clustering", page_icon="📊", layout="wide")
inject_css()

# --- Pill title banner ---
st.markdown("""
<div class="gradient-header">
<h1>Embedding, Clustering and Visualization Pipeline</h1>
<p>Upload CSV, run each step, or use your own outputs. Settings in the sidebar.</p>
</div>
""", unsafe_allow_html=True)
if "last_stdout" not in st.session_state:
    st.session_state.last_stdout = {}
if "last_stderr" not in st.session_state:
    st.session_state.last_stderr = {}

# --- Sidebar: collapsible expanders in step order ---
config = load_config()
with st.sidebar:
    st.subheader("Settings")

    with st.expander("Embedding", expanded=False):
        embed_test = st.number_input("Test rows (0 = all)", min_value=0, value=0, key="et")
        embed_skip = st.number_input("Skip first N rows", min_value=0, value=0, key="embed_skip")
        embed_workers = st.number_input("Workers", min_value=1, max_value=12, value=6, key="ew")
        embed_sequential = st.checkbox("Sequential (no parallel)", value=False, key="embed_sequential")

    with st.expander("Clustering", expanded=False):
        algorithm = st.selectbox("Algorithm", ["hdbscan", "kmeans", "agglomerative"], key="algo")
        k_val = st.number_input("K (kmeans/agglomerative)", min_value=2, value=10, key="k_val")
        min_cluster_size = st.number_input("Min cluster size", min_value=1, value=5, key="min_cluster_size")
        dimensions = st.number_input("Dimensions (dim reduction)", min_value=2, value=250, key="dimensions")
        assign_noise = st.checkbox("Assign noise to nearest cluster", value=False, key="assign_noise")
        no_dim_reduction = st.checkbox("No dimension reduction", value=False, key="no_dim_reduction")
        use_llm = st.checkbox("Use LLM for cluster labels", value=True, key="use_llm")
        use_sentiment = st.checkbox("Sentiment-aided clustering", value=True, key="use_sentiment")

    with st.expander("Sentiment words", expanded=False):
        neg = config.get("sentiment_words", {}).get("negative", [])
        pos = config.get("sentiment_words", {}).get("positive", [])
        neg_text = st.text_area("Negative (comma or newline)", value="\n".join(neg) if isinstance(neg, list) else str(neg), height=120, key="neg_words")
        pos_text = st.text_area("Positive (comma or newline)", value="\n".join(pos) if isinstance(pos, list) else str(pos), height=120, key="pos_words")

    with st.expander("Visualization", expanded=False):
        compare_only = st.checkbox("Compare only", value=False, key="compare_only")
        confirm = st.checkbox("Confirm", value=False, key="confirm")

    with st.expander("Color and chart", expanded=False):
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

    if st.button("Save config (structured)", key="save_structured"):
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
text_column_override = None
out_files = list(WORKSPACE.iterdir()) if WORKSPACE.exists() else []
out_files = [f for f in out_files if f.is_file()]
all_names = {f.name for f in out_files}
step_labels = {1: "Embedding", 2: "Clustering", 3: "Visualization"}


def _render_step_download(step: int):
    """Right column: one zip download + file names list for this step."""
    step_files = _files_for_step(step, all_names)
    label = step_labels[step]
    st.markdown(f"**Download {label} outputs**")
    if not step_files:
        st.caption(f"Run {label} to generate outputs.")
        return
    paths = [WORKSPACE / n for n in step_files]
    zip_bytes = _make_zip_bytes(paths)
    st.download_button(
        f"📥 Download all ({len(paths)} file{'s' if len(paths) != 1 else ''})",
        data=zip_bytes,
        file_name=f"step_{step}_outputs.zip",
        mime="application/zip",
        key=f"dl_step_{step}_zip",
    )
    for n in sorted(step_files):
        st.caption(f"  • {n}")

col_emb_left, col_emb_right = st.columns([1, 1])
with col_emb_left:
    st.markdown("**► Embedding**")
    can_embed = (WORKSPACE / INPUT_CSV).exists()
    upload_col, icon_col = st.columns([6, 1])
    with upload_col:
        uploaded = st.file_uploader("", type=["csv"], key="csv_upload", label_visibility="collapsed")
    with icon_col:
        st.markdown(
            '<span class="csv-upload-info" title="Limit 200MB per file • CSV">ⓘ</span>',
            unsafe_allow_html=True,
        )
    if uploaded is not None:
        df_preview = pd.read_csv(uploaded, nrows=5)
        all_cols = list(df_preview.columns)
        uploaded.seek(0)
        string_cols = get_string_columns(pd.read_csv(uploaded, nrows=500))
        if not string_cols:
            string_cols = all_cols
        path = WORKSPACE / INPUT_CSV
        with open(path, "wb") as f:
            f.write(uploaded.getvalue())
        can_embed = True
        with st.expander("Preview & text column", expanded=False):
            st.dataframe(pd.DataFrame(df_preview), use_container_width=True)
            if len(string_cols) > 1:
                text_column_override = st.selectbox("Text column", options=string_cols, key="text_col_select")
            else:
                text_column_override = string_cols[0] if string_cols else all_cols[0]
        st.caption(f"Saved as `{INPUT_CSV}`.")
    elif can_embed:
        try:
            df_ws = pd.read_csv(WORKSPACE / INPUT_CSV, nrows=500)
            string_cols = get_string_columns(df_ws)
            if not string_cols:
                string_cols = list(df_ws.columns)
            with st.expander("Preview & text column", expanded=False):
                if len(string_cols) >= 1:
                    text_column_override = st.selectbox("Text column (existing file)", options=string_cols, key="text_col_ws")
        except Exception:
            pass
    if not can_embed:
        st.caption("Upload a CSV to run Embedding.")
    if st.button("Run Embedding", disabled=not can_embed, key="btn_embed"):
        args = ["--input", str(WORKSPACE / INPUT_CSV)]
        if text_column_override:
            args += ["--text-column", text_column_override]
        args += ["--workers", str(st.session_state.get("ew", 6))]
        if st.session_state.get("et", 0) > 0:
            args += ["--test", str(st.session_state.et)]
        if st.session_state.get("embed_skip", 0) > 0:
            args += ["--skip", str(st.session_state.embed_skip)]
        if st.session_state.get("embed_sequential", False):
            args.append("--sequential")
        code, out, err = run_embedding_with_progress(SCRIPT_01, args, WORKSPACE)
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
with col_emb_right:
    _render_step_download(1)
st.divider()
col_cl_left, col_cl_right = st.columns([1, 1])
with col_cl_left:
    st.markdown("**► Clustering**")
    with st.expander("Use your own embedding (optional)", expanded=False):
        emb_npy = st.file_uploader("embeddings.npy", type=["npy"], key="up_emb_npy")
        emb_meta = st.file_uploader("embeddings_metadata.csv", type=["csv"], key="up_emb_meta")
        if emb_npy and emb_meta:
            with open(WORKSPACE / EMB_NPY, "wb") as f:
                f.write(emb_npy.getvalue())
            emb_meta_df = pd.read_csv(emb_meta)
            emb_meta_df.to_csv(WORKSPACE / EMB_META, index=False, encoding="utf-8-sig")
            st.caption("Saved. You can run Clustering.")
    can_cluster = workspace_has(EMB_NPY, EMB_META)
    if not can_cluster:
        st.caption("Run Embedding or upload embeddings above.")
    if st.button("Run Clustering", disabled=not can_cluster, key="btn_cluster"):
        args = ["--algorithm", st.session_state.get("algo", "hdbscan")]
        if not st.session_state.get("use_llm", True):
            args.append("--no-llm")
        if st.session_state.get("assign_noise", False):
            args.append("--assign-noise")
        if not st.session_state.get("use_sentiment", True):
            args.append("--no-sentiment")
        if st.session_state.get("algo", "hdbscan") in ("kmeans", "agglomerative"):
            args += ["--k", str(st.session_state.get("k_val", 10))]
        args += ["--min-cluster-size", str(st.session_state.get("min_cluster_size", 5))]
        args += ["--dimensions", str(st.session_state.get("dimensions", 250))]
        if st.session_state.get("no_dim_reduction", False):
            args.append("--no-dim-reduction")
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
with col_cl_right:
    _render_step_download(2)
st.divider()
col_viz_left, col_viz_right = st.columns([1, 1])
with col_viz_left:
    st.markdown("**► Visualization**")
    with st.expander("Use your own clusters (optional)", expanded=False):
        up_clusters = st.file_uploader("clusters_with_labels.csv", type=["csv"], key="up_clusters")
        up_meta_cl = st.file_uploader("metadata_with_clusters.csv", type=["csv"], key="up_meta_cl")
        if up_clusters and up_meta_cl:
            pd.read_csv(up_clusters).to_csv(WORKSPACE / CLUSTERS_CSV, index=False, encoding="utf-8-sig")
            pd.read_csv(up_meta_cl).to_csv(WORKSPACE / META_CLUSTERS_CSV, index=False, encoding="utf-8-sig")
            st.caption("Saved. You can run Visualization.")
        up_emb_meta_viz = st.file_uploader("embeddings_metadata.csv (for word clouds)", type=["csv"], key="up_emb_meta_viz")
        if up_emb_meta_viz:
            pd.read_csv(up_emb_meta_viz).to_csv(WORKSPACE / EMB_META, index=False, encoding="utf-8-sig")
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
    use_llm_viz = st.checkbox("LLM mode", value=False, key="use_llm_viz")
    if st.button("Run Visualization", disabled=not can_viz, key="btn_viz"):
        args = []
        if group_by_col and group_by_col != "Auto":
            args += ["--group-by", group_by_col]
        if groups_text.strip():
            args += ["--groups"] + groups_text.strip().split()
        if use_llm_viz:
            args.append("--llm")
        if st.session_state.get("compare_only", False):
            args.append("--compare-only")
        if st.session_state.get("confirm", False):
            args.append("--confirm")
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
with col_viz_right:
    _render_step_download(3)
