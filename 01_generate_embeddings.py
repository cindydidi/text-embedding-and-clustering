#!/usr/bin/env python3
"""
Step 1: Generate embeddings for text from a CSV using Google's Gemini Embedding API
"""

import sys
import csv
import os
import numpy as np
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError as e:
    print(f"\n❌ Import Error: {e}")
    print("\nPlease install missing packages:")
    print("  python3 -m pip install google-generativeai numpy pandas python-dotenv")
    sys.exit(1)

# Load environment variables from .env file
load_dotenv()

# Column name for the embedded text in output metadata (used by steps 02 and 03)
TEXT_COLUMN = "text"

def configure_api():
    """Configure Google API with key from environment."""
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment. Please set it in .env file or export it.")
    genai.configure(api_key=api_key)
    print("✓ Google API configured successfully")

# Global cache for embeddings (text -> embedding)
_embedding_cache = {}
# Global variable to store detected embedding dimension
_embedding_dimension = None

def generate_embedding(text, model_name='models/gemini-embedding-001', max_retries=3, use_cache=True):
    """
    Generate embedding for a single text using Google's API.
    Uses gemini-embedding-001 (better performance) by default.
    Caches results to avoid duplicate API calls for same text.
    
    Args:
        text: Text to embed
        model_name: Model to use
        max_retries: Maximum retry attempts
        use_cache: Whether to use cache (default True)
    """
    # Check cache first
    if use_cache and text in _embedding_cache:
        return _embedding_cache[text]
    
    # Generate new embedding
    for attempt in range(max_retries):
        try:
            result = genai.embed_content(
                model=model_name,
                content=text,
                task_type="RETRIEVAL_DOCUMENT"
            )
            embedding = result['embedding']
            
            # Detect and store embedding dimension from first successful call
            global _embedding_dimension
            if _embedding_dimension is None and embedding is not None:
                _embedding_dimension = len(embedding)
            
            # Cache the result
            if use_cache:
                _embedding_cache[text] = embedding
            
            return embedding
        except Exception as e:
            if attempt < max_retries - 1:
                if 'gemini-embedding-001' in model_name:
                    # Try text-embedding-004 as fallback
                    model_name = 'models/text-embedding-004'
                    time.sleep(0.5)
                    continue
                time.sleep(1 * (attempt + 1))  # Short backoff
            else:
                raise Exception(f"Failed to generate embedding after {max_retries} attempts: {e}")

def process_messages_parallel(texts, max_workers=6):
    """Process text rows in parallel using ThreadPoolExecutor.
    Optimized to reduce lock contention and improve speed.
    
    Args:
        texts: List of text strings to embed
        max_workers: Number of concurrent threads (default 6, optimized for API limits)
    
    Returns:
        List of embeddings in same order as texts
    """
    embeddings = [None] * len(texts)  # Pre-allocate to maintain order
    completed_count = 0
    failed_count = 0
    lock = Lock()
    start_time = time.time()
    last_progress_time = start_time
    
    def process_single(index, text):
        """Process a single text and store result at correct index."""
        nonlocal completed_count, failed_count
        try:
            embedding = generate_embedding(text, use_cache=True)
            embeddings[index] = embedding
            # Increment counter with minimal lock time
            with lock:
                completed_count += 1
            return index, True, None
        except Exception as e:
            # Get embedding dimension for placeholder (use detected or default)
            global _embedding_dimension
            dim = _embedding_dimension if _embedding_dimension is not None else 3072
            embeddings[index] = [0.0] * dim
            with lock:
                completed_count += 1
                failed_count += 1
            return index, False, str(e)
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(process_single, i, t): i 
            for i, t in enumerate(texts)
        }
        
        # Process completed tasks and show progress
        for future in as_completed(future_to_index):
            index, success, error = future.result()
            
            # Progress update (every 100 completions or every 2 seconds, or at end)
            # Check outside lock first to minimize contention
            current_time = time.time()
            should_update = False
            current_count = 0
            current_failed = 0
            
            with lock:
                current_count = completed_count
                current_failed = failed_count
                # Update less frequently: every 100 or every 2 seconds
                if (current_count % 100 == 0) or (current_time - last_progress_time >= 2.0) or (current_count == len(texts)):
                    should_update = True
                    if should_update:
                        last_progress_time = current_time
            
            # Print outside lock to avoid blocking threads
            if should_update:
                elapsed = current_time - start_time
                progress = (current_count / len(texts)) * 100
                rate = current_count / elapsed if elapsed > 0 else 0
                eta = (len(texts) - current_count) / rate if rate > 0 else 0
                
                print(f"\rProgress: {progress:.1f}% ({current_count:,}/{len(texts):,}) | "
                      f"Rate: {rate:.1f} text/s | ETA: {eta/60:.1f} min | "
                      f"Failed: {current_failed}", end='', flush=True)
    
    print()  # New line after progress
    if failed_count > 0:
        print(f"⚠ Warning: {failed_count} embeddings failed and were replaced with zero vectors")
    
    return embeddings

def process_messages_sequential(texts):
    """Process text rows sequentially (original faster version).
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        List of embeddings in same order as texts
    """
    embeddings = []
    start_time = time.time()
    total = len(texts)
    
    for i, text in enumerate(texts):
        try:
            embedding = generate_embedding(text, use_cache=True)
            embeddings.append(embedding)
            
            # Progress update every 50 rows
            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                progress = ((i + 1) / total) * 100
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (total - i - 1) / rate if rate > 0 else 0
                
                print(f"\rProgress: {progress:.1f}% ({i+1:,}/{total:,}) | "
                      f"Rate: {rate:.1f} text/s | ETA: {eta/60:.1f} min", end='', flush=True)
            
            # Minimal rate limiting
            if (i + 1) % 200 == 0:
                time.sleep(0.05)
                
        except Exception as e:
            # Get embedding dimension for placeholder
            global _embedding_dimension
            dim = _embedding_dimension if _embedding_dimension is not None else 3072
            embeddings.append([0.0] * dim)
            print(f"\nError processing text row {i+1}: {e}")
    
    print()  # New line after progress
    return embeddings

def _is_string_column(values):
    """Return True if column has any non-empty value that is not purely numeric."""
    for v in values:
        v = (v or '').strip()
        if not v:
            continue
        try:
            float(v)
        except ValueError:
            return True
    return False


def detect_text_column(input_file, sample_size=500):
    """
    Detect which CSV column to use for embeddings.
    - If CSV has only 1 column, use it.
    - If CSV has exactly 1 column that looks like text (non-numeric), use it.
    - If CSV has multiple text columns, raise ValueError (user must specify --text-column).
    """
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"The CSV file {input_file} has no header row or is empty.")
        sample_rows = []
        for i, row in enumerate(reader):
            if i >= sample_size:
                break
            sample_rows.append(row)

    if len(fieldnames) == 1:
        return fieldnames[0]

    string_columns = []
    for col in fieldnames:
        values = [row.get(col, '') for row in sample_rows]
        if _is_string_column(values):
            string_columns.append(col)

    if len(string_columns) == 1:
        return string_columns[0]
    if len(string_columns) > 1:
        raise ValueError(
            f"The CSV has multiple text columns: {string_columns}. "
            "Please specify which column to use for embeddings with: --text-column COLUMN_NAME"
        )
    raise ValueError(
        f"No text column found in CSV (columns look numeric or empty). Columns: {fieldnames}. "
        "Please specify the column to use with: --text-column COLUMN_NAME"
    )


def process_messages(input_file, output_embeddings, output_metadata, max_messages=None, skip_messages=0, max_workers=6, use_sequential=False, text_column=None):
    """Process text from CSV and generate embeddings with parallel or sequential processing.
    
    Args:
        input_file: Path to input CSV file
        output_embeddings: Path to save embeddings
        output_metadata: Path to save metadata
        max_messages: Maximum number of rows to process (None for all)
        skip_messages: Number of rows to skip at the start (default 0)
        max_workers: Number of concurrent threads (default 6, optimized for API limits)
        use_sequential: If True, use sequential processing (faster for some cases)
        text_column: CSV column name containing the text to embed (set by CLI or auto-detection)
    """
    if text_column is None:
        raise ValueError("text_column is required (set by --text-column or auto-detection in __main__)")

    function_start_time = time.time()
    
    print(f"Reading from {input_file} (text column: {text_column})...")
    if skip_messages > 0:
        print(f"  Skipping first {skip_messages:,} rows")
    if max_messages:
        print(f"  TEST MODE: Processing {max_messages:,} rows")
        if skip_messages > 0:
            print(f"  (Rows {skip_messages+1:,} to {skip_messages+max_messages:,})")
    else:
        print("  (This may take 10-30 seconds for large files)")
    texts = []
    metadata = []
    
    read_start = time.time()
    skipped_count = 0
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        reader = csv.DictReader(f)
        row_count = 0
        for row in reader:
            text = row.get(text_column, '').strip()
            if text:  # Only count non-empty text
                # Skip rows if we haven't reached skip_messages yet
                if skipped_count < skip_messages:
                    skipped_count += 1
                    continue
                
                # Stop if we've reached max_messages
                if max_messages and len(texts) >= max_messages:
                    break
                
                texts.append(text)
                # Pass through all columns from the input row; store embedded text for downstream scripts
                metadata.append({**row, TEXT_COLUMN: text})
            
            row_count += 1
            if row_count % 10000 == 0:
                print(f"  Read {row_count:,} rows, collected {len(texts):,} text rows...", end='\r', flush=True)
    
    read_time = time.time() - read_start
    total_texts = len(texts)
    print(f"\n✓ Found {total_texts:,} text rows to process (read in {read_time:.1f}s)")
    
    # Generate embeddings
    embedding_start_time = time.time()
    
    if use_sequential:
        print(f"\nGenerating embeddings sequentially for {total_texts:,} rows...")
        if max_messages:
            estimated_time = max_messages / 1.5  # Assume ~1.5 text/s
            print(f"Estimated time: ~{estimated_time:.1f} seconds ({estimated_time/60:.1f} minutes)")
        print("Progress: 0%", end='', flush=True)
        embeddings = process_messages_sequential(texts)
    else:
        if max_messages:
            estimated_time = max_messages / (max_workers * 1.5)  # Rough estimate with parallel
            print(f"\nGenerating embeddings for {total_texts:,} rows...")
            print(f"Using {max_workers} parallel workers")
            print(f"Estimated time: ~{estimated_time:.1f} seconds ({estimated_time/60:.1f} minutes)")
        else:
            print(f"\nGenerating embeddings with {max_workers} parallel workers...")
        print("Progress: 0%", end='', flush=True)
        embeddings = process_messages_parallel(texts, max_workers=max_workers)
    
    embedding_end_time = time.time()
    embedding_duration = embedding_end_time - embedding_start_time
    
    print(f"\n\n✓ Generated {len(embeddings):,} embeddings")
    print(f"⏱️  Time taken: {embedding_duration:.1f} seconds ({embedding_duration/60:.1f} minutes)")
    if len(embeddings) > 0:
        print(f"   Average: {embedding_duration/len(embeddings):.3f} seconds per row")
        print(f"   Rate: {len(embeddings)/embedding_duration:.1f} rows/second")
    
    # Convert to numpy array
    save_start_time = time.time()
    embeddings_array = np.array(embeddings)
    print(f"Embedding shape: {embeddings_array.shape}")
    
    # Check if output file already exists and merge if needed
    if os.path.exists(output_embeddings):
        print(f"\n⚠ Found existing file: {output_embeddings}")
        print("  Merging new embeddings with existing ones...")
        existing_embeddings = np.load(output_embeddings)
        print(f"  Existing: {existing_embeddings.shape[0]:,} embeddings")
        print(f"  New: {embeddings_array.shape[0]:,} embeddings")
        
        # Merge embeddings
        merged_embeddings = np.vstack([existing_embeddings, embeddings_array])
        print(f"  Merged: {merged_embeddings.shape[0]:,} total embeddings")
        embeddings_array = merged_embeddings
        
        # Also merge metadata if it exists
        if os.path.exists(output_metadata):
            existing_metadata = pd.read_csv(output_metadata)
            print(f"  Merging metadata: {len(existing_metadata):,} + {len(metadata):,} = {len(existing_metadata) + len(metadata):,}")
            metadata_df = pd.concat([existing_metadata, pd.DataFrame(metadata)], ignore_index=True)
        else:
            metadata_df = pd.DataFrame(metadata)
    else:
        metadata_df = pd.DataFrame(metadata)
    
    # Save embeddings
    np.save(output_embeddings, embeddings_array)
    save_end_time = time.time()
    save_duration = save_end_time - save_start_time
    
    print(f"✓ Saved {embeddings_array.shape[0]:,} embeddings to {output_embeddings} ({save_duration:.1f}s)")
    
    # Save metadata
    metadata_df.to_csv(output_metadata, index=False, encoding="utf-8-sig")
    print(f"✓ Saved {len(metadata_df):,} metadata rows to {output_metadata}")
    
    # Total time summary
    total_time = time.time() - function_start_time
    print(f"\n⏱️  TOTAL TIME: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"   - CSV reading: {read_time:.1f}s ({read_time/total_time*100:.1f}%)")
    print(f"   - Embedding generation: {embedding_duration:.1f}s ({embedding_duration/total_time*100:.1f}%)")
    print(f"   - File operations: {save_duration:.1f}s ({save_duration/total_time*100:.1f}%)")
    
    return embeddings_array, metadata_df

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate embeddings for text from a CSV')
    parser.add_argument('--input', '-i', type=str, default=None,
                       help='CSV file to use. Only needed when the folder contains more than one CSV.')
    parser.add_argument('--test', type=int, default=None,
                       help='Test mode: process only N rows (e.g., --test 1000)')
    parser.add_argument('--skip', type=int, default=0,
                       help='Skip first N rows (e.g., --skip 1000 to process rows 1001-2000)')
    parser.add_argument('--workers', type=int, default=6,
                       help='Number of parallel workers (default: 6, use --sequential for sequential mode)')
    parser.add_argument('--sequential', action='store_true',
                       help='Use sequential processing instead of parallel (may be faster)')
    parser.add_argument('--text-column', type=str, default=None,
                       help='CSV column name containing the text to embed. Required if CSV has multiple text columns.')
    args = parser.parse_args()
    
    # Auto-detect input CSV: use the one in the same folder as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
    if args.input is not None:
        input_file = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
        if not os.path.isfile(input_file):
            print(f"Error: File not found: {input_file}")
            exit(1)
    elif len(csv_files) == 0:
        print("No CSV file found in this folder. Put your CSV file in the same folder as this script, then run again.")
        exit(1)
    elif len(csv_files) == 1:
        input_file = os.path.join(script_dir, csv_files[0])
        print(f"Using CSV: {csv_files[0]}")
    else:
        print(f"Multiple CSV files found: {', '.join(sorted(csv_files))}")
        print("Please specify which to use with: --input filename.csv")
        exit(1)
    
    print("="*60)
    print("Starting Embedding Generation")
    if args.skip > 0:
        print(f"Skipping first {args.skip:,} rows")
    if args.test:
        start_row = args.skip + 1
        end_row = args.skip + args.test
        print(f"TEST MODE: Processing {args.test:,} rows (rows {start_row:,} to {end_row:,})")
    print("="*60)
    
    output_embeddings = 'embeddings.npy'
    output_metadata = 'embeddings_metadata.csv'
    
    # Resolve which CSV column to use for text
    if args.text_column is not None:
        # Validate that the column exists in the CSV
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
        if args.text_column not in fieldnames:
            print(f"Error: Column '{args.text_column}' not found in {input_file}.")
            print(f"Available columns: {fieldnames}")
            exit(1)
        text_column = args.text_column
        print(f"Using text column: {text_column}")
    else:
        try:
            text_column = detect_text_column(input_file)
            print(f"Using text column: {text_column}")
        except ValueError as e:
            print(f"\n{e}")
            exit(1)
    
    # Add test suffix to output files if in test mode
    if args.test:
        # Always use the same test file names so we can merge
        output_embeddings = 'embeddings_test.npy'
        output_metadata = 'embeddings_metadata_test.csv'
    
    try:
        configure_api()
        if args.sequential:
            print("Using sequential processing mode")
        else:
            print(f"Using {args.workers} parallel workers for faster processing")
        embeddings, metadata = process_messages(input_file, output_embeddings, output_metadata, 
                                                max_messages=args.test, skip_messages=args.skip, 
                                                max_workers=args.workers, use_sequential=args.sequential,
                                                text_column=text_column)
        print("\n" + "="*60)
        print("Step 1 Complete! Embeddings generated successfully.")
        print("="*60)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
