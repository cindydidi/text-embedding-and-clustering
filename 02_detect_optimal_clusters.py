#!/usr/bin/env python3
"""
Step 2: Auto-detect optimal number of clusters using Elbow Method and Silhouette Score
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

def load_embeddings(embeddings_file):
    """Load embeddings from previous step."""
    print(f"Loading embeddings from {embeddings_file}...")
    embeddings = np.load(embeddings_file)
    print(f"✓ Loaded {embeddings.shape[0]:,} embeddings with dimension {embeddings.shape[1]}")
    return embeddings

def find_optimal_clusters(embeddings, min_k=5, max_k=30, sample_size=None):
    """
    Find optimal number of clusters using Elbow Method and Silhouette Score.
    
    Args:
        embeddings: numpy array of embeddings
        min_k: minimum number of clusters to test
        max_k: maximum number of clusters to test
        sample_size: if provided, sample this many embeddings for faster computation
    """
    print(f"\nTesting cluster numbers from {min_k} to {max_k}...")
    
    # Sample embeddings if dataset is large (for faster computation)
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
        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        kmeans.fit(sample_embeddings)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        inertias.append(inertia)
        
        # Silhouette score (can be slow for large datasets)
        if len(sample_embeddings) < 10000:
            silhouette = silhouette_score(sample_embeddings, kmeans.labels_)
            silhouette_scores.append(silhouette)
        else:
            # For very large datasets, sample for silhouette
            sample_for_silhouette = min(5000, len(sample_embeddings))
            indices = np.random.choice(len(sample_embeddings), sample_for_silhouette, replace=False)
            silhouette = silhouette_score(sample_embeddings[indices], kmeans.labels_[indices])
            silhouette_scores.append(silhouette)
        
        progress = ((i + 1) / len(k_range)) * 100
        print(f"\rProgress: {progress:.1f}% (k={k})", end='', flush=True)
    
    print("\n✓ Completed cluster testing")
    
    # Find optimal k using elbow method (find the "knee" of the curve)
    # Calculate rate of change in inertia
    if len(inertias) > 1:
        # Normalize inertias for comparison
        normalized_inertias = [(x - min(inertias)) / (max(inertias) - min(inertias)) for x in inertias]
        # Calculate second derivative to find elbow
        second_deriv = []
        for i in range(1, len(normalized_inertias) - 1):
            second_deriv.append(normalized_inertias[i-1] - 2*normalized_inertias[i] + normalized_inertias[i+1])
        
        # Elbow is where second derivative is maximum
        if second_deriv:
            elbow_idx = np.argmax(second_deriv) + 1  # +1 because we start from index 1
            elbow_k = k_range[elbow_idx]
        else:
            elbow_k = k_range[len(k_range) // 2]
    else:
        elbow_k = min_k
    
    # Also consider silhouette score (higher is better)
    best_silhouette_idx = np.argmax(silhouette_scores)
    best_silhouette_k = k_range[best_silhouette_idx]
    
    # Choose optimal k (prefer silhouette if it's reasonable, otherwise use elbow)
    if silhouette_scores[best_silhouette_idx] > 0.3:  # Good silhouette score threshold
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

def plot_results(results, output_file):
    """Create visualization of cluster analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    k_range = results['k_range']
    inertias = results['inertias']
    silhouette_scores = results['silhouette_scores']
    optimal_k = results['optimal_k']
    
    # Plot 1: Elbow Method
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    ax1.set_title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Silhouette Score
    ax2.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'Optimal k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Silhouette Score', fontsize=12)
    ax2.set_title('Silhouette Score for Optimal k', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_file}")

def save_optimal_k(optimal_k, output_file):
    """Save optimal k to a text file."""
    with open(output_file, 'w') as f:
        f.write(str(optimal_k))
    print(f"✓ Saved optimal k={optimal_k} to {output_file}")

if __name__ == '__main__':
    embeddings_file = 'embeddings.npy'
    output_plot = 'optimal_clusters.png'
    output_k = 'optimal_k.txt'
    
    try:
        # Load embeddings
        embeddings = load_embeddings(embeddings_file)
        
        # Find optimal clusters
        # For large datasets, sample for faster computation
        sample_size = 10000 if len(embeddings) > 10000 else None
        results = find_optimal_clusters(embeddings, min_k=5, max_k=30, sample_size=sample_size)
        
        # Print results
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS RESULTS")
        print("="*60)
        print(f"Optimal number of clusters: {results['optimal_k']} (method: {results['method']})")
        print(f"Elbow method suggests: k={results['elbow_k']}")
        print(f"Best silhouette score: k={results['best_silhouette_k']} "
              f"(score: {results['silhouette_scores'][results['best_silhouette_k'] - results['k_range'][0]]:.3f})")
        print("="*60)
        
        # Save results
        plot_results(results, output_plot)
        save_optimal_k(results['optimal_k'], output_k)
        
        print("\n" + "="*60)
        print("Step 2 Complete! Optimal clusters detected.")
        print("="*60)
        
    except FileNotFoundError:
        print(f"Error: {embeddings_file} not found. Please run Step 1 first.")
        exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
