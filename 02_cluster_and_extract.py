#!/usr/bin/env python3
"""
Step 2: Perform clustering and extract cluster labels
(dimensionality reduction + optional LLM-based labels).
For HDBSCAN (default) no K is needed. For kmeans/agglomerative, K comes from --k,
optimal_k.txt, or automatic detection (elbow + silhouette).
"""

import json
import numpy as np
import pandas as pd
import argparse
import os
import time
from pathlib import Path
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError:
    print("Warning: google-generativeai not installed. LLM labels will use fallback.")
    genai = None

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    print("Warning: hdbscan not installed. Install with: pip install hdbscan")
    HDBSCAN_AVAILABLE = False
    hdbscan = None

# Load environment variables
load_dotenv()

# Column name for text content in metadata (must match step 01 output)
TEXT_COLUMN = "text"


def _text_column(metadata):
    """Return the text column name (prefer TEXT_COLUMN, fallback to 'Message' for old outputs)."""
    if TEXT_COLUMN in metadata.columns:
        return TEXT_COLUMN
    if "Message" in metadata.columns:
        return "Message"
    return None


# Global cache for LLM-generated labels
_llm_label_cache = {}

def configure_api():
    """Configure Google API with key from environment."""
    if genai is None:
        return False
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        print("Warning: GOOGLE_API_KEY not found. LLM labels will use fallback.")
        return False
    genai.configure(api_key=api_key)
    return True

def reduce_dimensions(embeddings, n_components=250):
    """
    Reduce embeddings using PCA to improve clustering performance.
    
    Args:
        embeddings: numpy array of embeddings (n_samples, n_features)
        n_components: number of dimensions to reduce to (default 250)
    
    Returns:
        reduced_embeddings: numpy array of reduced embeddings
        pca: fitted PCA object
    """
    print(f"\nReducing dimensions from {embeddings.shape[1]}D to {n_components}D using PCA...")
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"✓ Explained variance: {explained_variance:.2%}")
    return reduced, pca


def find_optimal_clusters(embeddings, min_k=5, max_k=30, sample_size=None):
    """
    Find optimal number of clusters using Elbow Method and Silhouette Score.
    Used when running kmeans/agglomerative without --k or optimal_k.txt.
    """
    print(f"\nTesting cluster numbers from {min_k} to {max_k}...")
    if sample_size and len(embeddings) > sample_size:
        print(f"Sampling {sample_size:,} embeddings for faster computation...")
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
    else:
        sample_embeddings = embeddings

    k_range = range(min_k, max_k + 1)
    inertias = []
    silhouette_scores = []
    print("Progress: 0%", end='', flush=True)
    for i, k in enumerate(k_range):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(sample_embeddings)
        inertias.append(kmeans.inertia_)
        if len(sample_embeddings) < 10000:
            silhouette_scores.append(silhouette_score(sample_embeddings, kmeans.labels_))
        else:
            n_s = min(5000, len(sample_embeddings))
            idx = np.random.choice(len(sample_embeddings), n_s, replace=False)
            silhouette_scores.append(silhouette_score(sample_embeddings[idx], kmeans.labels_[idx]))
        print(f"\rProgress: {((i + 1) / len(k_range)) * 100:.1f}% (k={k})", end='', flush=True)
    print("\n✓ Completed cluster testing")

    if len(inertias) > 1:
        norm = [(x - min(inertias)) / (max(inertias) - min(inertias)) for x in inertias]
        second_deriv = [norm[i-1] - 2*norm[i] + norm[i+1] for i in range(1, len(norm) - 1)]
        elbow_idx = np.argmax(second_deriv) + 1 if second_deriv else len(k_range) // 2
        elbow_k = list(k_range)[elbow_idx]
    else:
        elbow_k = min_k
    best_sil_idx = np.argmax(silhouette_scores)
    best_silhouette_k = list(k_range)[best_sil_idx]
    if silhouette_scores[best_sil_idx] > 0.3:
        optimal_k = best_silhouette_k
        method = "silhouette"
    else:
        optimal_k = elbow_k
        method = "elbow"
    return {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'optimal_k': optimal_k,
        'method': method,
        'elbow_k': elbow_k,
        'best_silhouette_k': best_silhouette_k
    }


def _plot_optimal_k_results(results, output_file):
    """Create elbow + silhouette visualization for optimal k."""
    k_range = results['k_range']
    inertias = results['inertias']
    silhouette_scores = results['silhouette_scores']
    optimal_k = results['optimal_k']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score for Optimal k', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved visualization to {output_file}")


def _save_optimal_k(optimal_k, output_file):
    """Save optimal k to a text file."""
    with open(output_file, 'w') as f:
        f.write(str(optimal_k))
    print(f"✓ Saved optimal k={optimal_k} to {output_file}")


def load_data(embeddings_file, metadata_file, optimal_k_file, algorithm, manual_k=None,
              optimal_k_plot='optimal_clusters.png', min_k=5, max_k=30):
    """
    Load embeddings and metadata. Resolve k for kmeans/agglomerative:
    1) manual_k (--k), 2) optimal_k.txt, 3) run find_optimal_clusters and save.
    For hdbscan, optimal_k is not read or used.
    """
    print("Loading data...")
    embeddings = np.load(embeddings_file)
    metadata = pd.read_csv(metadata_file)
    print(f"✓ Loaded {len(embeddings):,} embeddings")

    if algorithm == 'hdbscan':
        return embeddings, metadata, None

    if manual_k is not None:
        print(f"✓ Using manual k: {manual_k}")
        return embeddings, metadata, manual_k

    if os.path.isfile(optimal_k_file):
        with open(optimal_k_file, 'r') as f:
            optimal_k = int(f.read().strip())
        print(f"✓ Optimal k from file: {optimal_k}")
        return embeddings, metadata, optimal_k

    # Need k but no file and no --k: run optimal K detection
    print("No --k and no optimal_k.txt; running optimal K detection...")
    sample_size = 10000 if len(embeddings) > 10000 else None
    results = find_optimal_clusters(embeddings, min_k=min_k, max_k=max_k, sample_size=sample_size)
    _plot_optimal_k_results(results, optimal_k_plot)
    _save_optimal_k(results['optimal_k'], optimal_k_file)
    print(f"Optimal k: {results['optimal_k']} (method: {results['method']})")
    return embeddings, metadata, results['optimal_k']


# Sentiment lexicon: loaded from config/sentiment_words.json (longer matches first)
def _load_sentiment_config():
    config_path = Path(__file__).resolve().parent / "config" / "sentiment_words.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        neg = [s.strip() for s in data.get("negative", []) if s and isinstance(s, str)]
        pos = [s.strip() for s in data.get("positive", []) if s and isinstance(s, str)]
        return sorted(neg, key=len, reverse=True), sorted(pos, key=len, reverse=True)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: Could not load {config_path}: {e}. Using minimal sentiment lists.")
        return sorted(["bad"], key=len, reverse=True), sorted(["good"], key=len, reverse=True)


SENTIMENT_NEGATIVE, SENTIMENT_POSITIVE = _load_sentiment_config()


def _get_sentiment(text):
    """Return -1 (negative), 0 (neutral), or 1 (positive). Check negative first; longer matches first."""
    if not text or not isinstance(text, str):
        return 0
    t = text.strip()
    t_lower = t.lower()
    for token in SENTIMENT_NEGATIVE:
        if token in t or token in t_lower:
            return -1
    for token in SENTIMENT_POSITIVE:
        if token in t or token in t_lower:
            return 1
    return 0


def augment_embeddings_with_sentiment(embeddings, metadata, scale=0.5):
    """Append a sentiment dimension (-scale, 0, +scale) so clustering can separate by sentiment."""
    col = _text_column(metadata)
    if col is None:
        print("  Warning: No text column found for sentiment; skipping.")
        return embeddings
    sentiments = np.array([_get_sentiment(metadata[col].iloc[i]) for i in range(len(metadata))], dtype=np.float64)
    # Reshape (n,) -> (n, 1), scale so it has visible effect but doesn't dominate
    extra = (sentiments * scale).reshape(-1, 1)
    n_pos = (sentiments == 1).sum()
    n_neg = (sentiments == -1).sum()
    print(f"  Sentiment: {n_pos:,} positive, {n_neg:,} negative, {len(sentiments) - n_pos - n_neg:,} neutral")
    return np.hstack([embeddings, extra])


def _assign_noise_to_nearest_cluster(embeddings_normalized, cluster_labels):
    """Assign each point with label -1 to the cluster of its nearest non-noise neighbor."""
    noise_idx = np.where(cluster_labels == -1)[0]
    non_noise_idx = np.where(cluster_labels != -1)[0]
    if len(noise_idx) == 0 or len(non_noise_idx) == 0:
        return cluster_labels
    labels = np.asarray(cluster_labels, dtype=np.int64)
    # Distances from each noise point to each non-noise point
    D = pairwise_distances(embeddings_normalized[noise_idx], embeddings_normalized[non_noise_idx], metric='euclidean')
    nearest = np.argmin(D, axis=1)
    labels[noise_idx] = labels[non_noise_idx[nearest]]
    return labels


def perform_clustering(embeddings, k, algorithm='hdbscan', use_dim_reduction=True, n_components=250, min_cluster_size=None, assign_noise=False):
    """
    Perform clustering with specified algorithm.
    
    Args:
        embeddings: numpy array of embeddings
        k: number of clusters (used for kmeans/agglomerative, ignored for hdbscan)
        algorithm: 'hdbscan' (recommended), 'agglomerative', or 'kmeans'
        use_dim_reduction: whether to reduce dimensions before clustering
        n_components: number of dimensions if reducing (default 250)
        min_cluster_size: minimum cluster size for HDBSCAN (default: auto)
        assign_noise: if True, assign HDBSCAN noise (-1) to nearest cluster (default: False)
    """
    print(f"\nPerforming {algorithm} clustering...")
    
    # Reduce dimensions if requested
    if use_dim_reduction and embeddings.shape[1] > n_components:
        embeddings_reduced, _ = reduce_dimensions(embeddings, n_components)
        embeddings_to_cluster = embeddings_reduced
    else:
        embeddings_to_cluster = embeddings
    
    # Normalize embeddings (important for cosine similarity space)
    embeddings_normalized = normalize(embeddings_to_cluster)
    
    if algorithm == 'hdbscan':
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN not available. Install with: pip install hdbscan")
        
        # HDBSCAN: density-based clustering, automatically determines number of clusters
        print("  Using HDBSCAN (density-based, auto-detects clusters)...")
        
        # Auto-determine min_cluster_size if not provided (stricter = fewer, larger clusters)
        if min_cluster_size is None:
            # ~0.5% of data, at least 20 (larger = fewer clusters)
            min_cluster_size = max(100, int(len(embeddings) * 0.005))
            print(f"  Auto min_cluster_size: {min_cluster_size}")
        
        min_samp = min(5, max(1, min_cluster_size // 2))
        # 'eom' = fewer, more stable clusters; more points may be marked as noise (use --assign-noise if needed)
        clustering = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samp,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        cluster_labels = clustering.fit_predict(embeddings_normalized)
        
        n_noise = (cluster_labels == -1).sum()
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        print(f"✓ HDBSCAN clustering complete.")
        print(f"  Found {n_clusters} clusters")
        print(f"  Assigned {len(cluster_labels) - n_noise:,} messages to clusters")
        if n_noise > 0:
            print(f"  {n_noise:,} messages marked as noise (outliers)")
            if assign_noise:
                print(f"  Assigning noise points to nearest cluster...")
                cluster_labels = _assign_noise_to_nearest_cluster(embeddings_normalized, cluster_labels)
                print(f"  Done. All points now assigned to a cluster.")
        
        return cluster_labels, clustering
    
    elif algorithm == 'agglomerative':
        # Agglomerative clustering with cosine distance (better for embeddings)
        print(f"  Using cosine distance with k={k}...")
        # Use 'metric' instead of 'affinity' for newer scikit-learn versions
        try:
            # Try new API first (scikit-learn 1.2+)
            clustering = AgglomerativeClustering(
                n_clusters=k,
                metric='cosine',
                linkage='average'  # Average linkage works well with cosine distance
            )
        except TypeError:
            # Fallback to old API (older scikit-learn versions)
            clustering = AgglomerativeClustering(
                n_clusters=k,
                affinity='cosine',
                linkage='average'
            )
        cluster_labels = clustering.fit_predict(embeddings_normalized)
        print(f"✓ Agglomerative clustering complete. Assigned {len(cluster_labels):,} messages to {k} clusters")
        return cluster_labels, clustering
    
    elif algorithm == 'kmeans':
        # K-means clustering (original method)
        print(f"  Using K-means with k={k}...")
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        cluster_labels = kmeans.fit_predict(embeddings_normalized)
        print(f"✓ K-means clustering complete. Assigned {len(cluster_labels):,} messages to {k} clusters")
        return cluster_labels, kmeans
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'hdbscan', 'agglomerative', or 'kmeans'")

def select_representative_messages(messages, n=8):
    """
    Select diverse representative messages from a cluster.
    
    Args:
        messages: list of message strings
        n: number of messages to select
    
    Returns:
        list of selected messages
    """
    if len(messages) <= n:
        return messages
    
    # Filter messages by length (prefer 20-200 chars)
    filtered = [msg for msg in messages if 20 <= len(msg) <= 200]
    if len(filtered) < n:
        # If not enough filtered, use all messages
        filtered = messages
    
    # Select diverse samples: first, middle, last, and random
    selected = []
    indices_used = set()
    
    # First message
    if len(filtered) > 0:
        selected.append(filtered[0])
        indices_used.add(0)
    
    # Middle messages
    if len(filtered) > 2:
        mid = len(filtered) // 2
        if mid not in indices_used:
            selected.append(filtered[mid])
            indices_used.add(mid)
    
    # Last message
    if len(filtered) > 1:
        last_idx = len(filtered) - 1
        if last_idx not in indices_used:
            selected.append(filtered[last_idx])
            indices_used.add(last_idx)
    
    # Fill remaining with random unique messages
    import random
    remaining = [i for i in range(len(filtered)) if i not in indices_used]
    random.shuffle(remaining)
    
    while len(selected) < n and remaining:
        idx = remaining.pop()
        selected.append(filtered[idx])
    
    return selected[:n]

def generate_cluster_label_llm(messages, cluster_id):
    """
    Generate cluster label using Gemini LLM API.
    Falls back to simple keyword extraction if API fails.
    
    Args:
        messages: list of messages in cluster
        cluster_id: cluster ID
    
    Returns:
        tuple: (label, keywords_list)
    """
    if len(messages) < 3:
        return f"Cluster {cluster_id+1}", []
    
    # Check cache (convert to str so mixed types e.g. str/float from DataFrame don't break sort)
    cache_key = hash(tuple(sorted(str(m) for m in messages[:10])))
    if cache_key in _llm_label_cache:
        return _llm_label_cache[cache_key]
    
    # Select representative messages
    sample_messages = select_representative_messages(messages, n=8)
    
    # Try LLM generation
    if genai is not None:
        try:
            # Configure API if not already done
            if not hasattr(genai, '_configured'):
                if configure_api():
                    genai._configured = True
                else:
                    genai._configured = False
            
            if genai._configured:
                # Create prompt
                messages_text = "\n".join([f"- {msg[:200]}" for msg in sample_messages[:8]])  # Limit message length
                prompt = f"""Generate a concise cluster label (2-4 words) for these customer service messages. 
The label should describe the main theme of the cluster.

Messages:
{messages_text}

Cluster label (2-4 words only):"""
                
                # Call Gemini API (try models in order; names vary by API version/region)
                model_names = ['gemini-2.0-flash', 'gemini-1.5-flash', 'gemini-2.5-pro']
                label = None
                for model_name in model_names:
                    try:
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(prompt)
                        label = response.text.strip()
                        break
                    except Exception:
                        continue
                if label:
                    # Clean up label (remove quotes, extra whitespace)
                    label = label.replace('"', '').replace("'", '').strip()
                    # Limit to 4 words max
                    words = label.split()[:4]
                    label = " ".join(words)
                    keywords = extract_keywords_simple(messages)
                    result = (label, keywords)
                    _llm_label_cache[cache_key] = result
                    return result
        except Exception as e:
            print(f"\nWarning: LLM label generation failed for cluster {cluster_id}: {e}")
            print("  Using fallback keyword extraction...")
    
    # Fallback to simple keyword extraction
    return generate_cluster_label_fallback(messages, cluster_id)

def extract_keywords_simple(messages):
    """Extract keywords from messages using simple frequency counting."""
    import re
    from collections import Counter
    
    # Common stopwords
    stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'can', 'i', 'you', 'he', 'she', 'it', 'we',
                     'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their', 'this', 'that',
                     'these', 'those', 'what', 'which', 'who', 'when', 'where', 'why', 'how',
                     'hi', 'hello', 'hey', 'please', 'thank', 'thanks', 'yes', 'no', 'ok',
                     'okay', 'need', 'want', 'get', 'got', 'go', 'went', 'come', 'came'])
    
    all_words = []
    for msg in messages:
        # Extract words (alphanumeric, at least 3 chars)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', msg.lower())
        words = [w for w in words if w not in stopwords]
        all_words.extend(words)
    
    # Count and return top keywords
    word_counts = Counter(all_words)
    top_keywords = [word for word, count in word_counts.most_common(5)]
    return top_keywords

def generate_cluster_label_fallback(messages, cluster_id):
    """Fallback cluster label generation using simple keyword extraction."""
    keywords = extract_keywords_simple(messages)
    if keywords:
        label = " / ".join(keywords[:3]).title()
    else:
        label = f"Cluster {cluster_id+1}"
    return label, keywords

def analyze_clusters(cluster_labels, metadata, use_llm=True):
    """Analyze clusters and extract cluster labels."""
    print("\nAnalyzing clusters and extracting labels...")
    metadata["Cluster"] = cluster_labels
    
    clusters_data = []
    used_labels = set()
    
    # Get unique cluster IDs (excluding noise label -1 if present)
    unique_clusters = sorted([c for c in set(cluster_labels) if c != -1])
    total_clusters = len(unique_clusters)
    
    # Handle noise points separately if they exist
    n_noise = (cluster_labels == -1).sum()
    if n_noise > 0:
        print(f"  Note: {n_noise:,} messages marked as noise (outliers) will be excluded from cluster analysis")
    
    # Discover category columns from data: any column (except text/Cluster) with 2–200 unique values
    text_col = _text_column(metadata)
    skip_cols = {"Cluster"}
    if text_col:
        skip_cols.add(text_col)
    category_columns = []
    for col in metadata.columns:
        if col in skip_cols:
            continue
        uniq = metadata[col].dropna().astype(str).str.strip()
        uniq = uniq[uniq != ""]
        n_uniq = uniq.nunique()
        if 2 <= n_uniq <= 200:
            vals = sorted(uniq.unique().tolist())
            category_columns.append((col, vals))
    
    start_time = time.time()
    
    for idx, cluster_id in enumerate(unique_clusters):
        text_col = _text_column(metadata)
        if not text_col:
            continue
        raw = metadata.loc[
            metadata["Cluster"] == cluster_id, text_col
        ].dropna().astype(str).str.strip()
        cluster_messages = [m for m in raw.tolist() if m]
        
        if not cluster_messages:
            continue
        
        # Generate cluster label
        if use_llm:
            cluster_label, top_keywords = generate_cluster_label_llm(cluster_messages, cluster_id)
        else:
            cluster_label, top_keywords = generate_cluster_label_fallback(cluster_messages, cluster_id)
        
        # Ensure label is unique
        original_label = cluster_label
        counter = 1
        while cluster_label.lower() in used_labels:
            cluster_label = f"{original_label} ({counter})"
            counter += 1
        
        used_labels.add(cluster_label.lower())
        
        cluster_metadata = metadata[metadata["Cluster"] == cluster_id]
        row_data = {
            "Cluster_ID": cluster_id,
            "Cluster_Label": cluster_label,
            "Top_Keywords": ", ".join(top_keywords),
            "Message_Count": len(cluster_messages),
            "Sample_Messages": " | ".join(cluster_messages[:5])
        }
        for col, values in category_columns:
            value_counts = cluster_metadata[col].value_counts()
            col_safe = col.replace(" ", "_")
            for val in values:
                safe_val = str(val).replace(" ", "_")
                # Use __ so 06 can parse column vs value (column__value_Count)
                row_data[f"{col_safe}__{safe_val}_Count"] = int(value_counts.get(val, 0))
        clusters_data.append(row_data)
        
        # Progress update
        progress = (idx + 1) / total_clusters * 100
        elapsed = time.time() - start_time
        rate = (idx + 1) / elapsed if elapsed > 0 else 0
        eta = (total_clusters - idx - 1) / rate if rate > 0 else 0
        print(f"\rProgress: {progress:.1f}% ({idx + 1}/{total_clusters}) | "
              f"ETA: {eta/60:.1f} min", end="", flush=True)
    
    print("\n✓ Cluster label extraction complete")
    return pd.DataFrame(clusters_data), metadata

def save_results(clusters_df, metadata_with_clusters, output_file):
    """Save clustering results."""
    clusters_df = clusters_df.sort_values("Message_Count", ascending=False)
    clusters_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    metadata_with_clusters.to_csv("metadata_with_clusters.csv", index=False, encoding="utf-8-sig")
    print(f"✓ Saved results to {output_file} and metadata_with_clusters.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cluster embeddings and extract cluster labels')
    parser.add_argument('--k', type=int, default=None,
                        help='Manual override for number of clusters (ignores optimal_k.txt, not used for HDBSCAN)')
    parser.add_argument('--algorithm', type=str, default='hdbscan',
                        choices=['hdbscan', 'agglomerative', 'kmeans'],
                        help='Clustering algorithm: hdbscan (recommended), agglomerative, or kmeans (default: hdbscan)')
    parser.add_argument('--min-cluster-size', type=int, default=None,
                        help='Minimum cluster size for HDBSCAN (default: auto, ~0.1%% of data)')
    parser.add_argument('--assign-noise', action='store_true',
                        help='Assign HDBSCAN noise points (-1) to nearest cluster so all points get a cluster')
    parser.add_argument('--no-dim-reduction', action='store_true',
                        help='Disable dimensionality reduction (use original embeddings)')
    parser.add_argument('--no-llm', action='store_true',
                        help='Disable LLM label generation (use fallback keyword extraction)')
    parser.add_argument('--no-sentiment', action='store_true',
                        help='Disable sentiment-aided clustering (default: add a sentiment dimension to reduce mixing positive/negative)')
    parser.add_argument('--dimensions', type=int, default=250,
                        help='Number of dimensions for PCA reduction (default: 250, higher preserves more cluster distinction)')
    
    args = parser.parse_args()
    
    embeddings_file = "embeddings.npy"
    metadata_file = "embeddings_metadata.csv"
    optimal_k_file = "optimal_k.txt"
    optimal_k_plot = "optimal_clusters.png"
    output_file = "clusters_with_labels.csv"

    # Load data; for HDBSCAN no k is needed; for kmeans/agglomerative k from --k, file, or auto-detect
    embeddings, metadata, optimal_k = load_data(
        embeddings_file, metadata_file, optimal_k_file,
        algorithm=args.algorithm,
        manual_k=args.k,
        optimal_k_plot=optimal_k_plot
    )
    
    # Optionally add sentiment as an extra dimension so positive/negative don't mix in one cluster
    if not args.no_sentiment:
        print("\nAdding sentiment dimension (lexicon-based)...")
        embeddings = augment_embeddings_with_sentiment(embeddings, metadata)
    
    # Perform clustering
    cluster_labels, clustering_model = perform_clustering(
        embeddings, 
        optimal_k,
        algorithm=args.algorithm,
        use_dim_reduction=not args.no_dim_reduction,
        n_components=args.dimensions,
        min_cluster_size=args.min_cluster_size,
        assign_noise=args.assign_noise
    )
    
    # Analyze clusters and extract labels
    clusters_df, metadata_with_clusters = analyze_clusters(
        cluster_labels, metadata, use_llm=not args.no_llm
    )
    
    # Save results
    save_results(clusters_df, metadata_with_clusters, output_file)
    
    print("\n" + "="*60)
    print("Step 2 Complete!")
    print("="*60)
    print(f"Generated {len(clusters_df)} clusters")
    print(f"Algorithm: {args.algorithm}")
    print(f"Used LLM labels: {not args.no_llm}")
    print(f"Sentiment-aided clustering: {not args.no_sentiment}")
    print(f"Used dimension reduction: {not args.no_dim_reduction}")
