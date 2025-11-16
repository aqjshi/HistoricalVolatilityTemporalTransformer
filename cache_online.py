import pandas as pd
import pyarrow 
import sys
import json
import re
import os
import requests 
import nltk
from nltk.corpus import words
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# --- 1. Setup NLTK and Word Lists ---
try:
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading 'words' corpus for dictionary check...")
    nltk.download('words')

print("Loading word lists into sets...")
ENGLISH_WORDS = set(words.words())
TICKER_SYMBOLS = {"meta", "amd", "aapl", "msft", "tsla", "nvda"}

try:
    with open("stopwords.json") as f:
        data = json.load(f)
    # Ensure it's a set for fast lookups
    HTML_ATTRIBUTE_STOP_WORDS = set(data['naive_stopwords'])
    print(f"Loaded {len(HTML_ATTRIBUTE_STOP_WORDS)} existing noise words from cache.")
except FileNotFoundError:
    print("Error: stopwords.json not found.")
    sys.exit(1)
except KeyError:
    print("Error: 'naive_stopwords' key not found in stopwords.json.")
    sys.exit(1)


# --- 2. Worker Functions ---

def fetch_url(idx, url):
    """
    Worker function for threading. Fetches URL content.
    Returns (idx, url, content_string, error_message)
    """
    if not isinstance(url, str) or not url.strip():
        return (idx, url, None, "Invalid URL (Not a string)")
    
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}
        response = requests.get(url, timeout=10, headers=headers)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        return (idx, url, response.text, None)
    
    except requests.exceptions.RequestException as e:
        return (idx, url, None, str(e))


def extract_potential_noise_words(raw_html, existing_stopwords):
    """
    Cleans HTML and returns a list of words that are *not* English, 
    *not* tickers, and *not* already in the existing_stopwords set.
    """
    
    # 1. STEP 1: Remove Style blocks and attributes
    style_block_pattern = re.compile(r'<style.*?>(.*?)</style>', re.DOTALL | re.IGNORECASE)
    intermediate_html = re.sub(style_block_pattern, ' ', raw_html)
    style_attr_pattern = re.compile(r'style=".*?"', re.IGNORECASE)
    intermediate_html = re.sub(style_attr_pattern, ' ', intermediate_html)
    
    # 2. STEP 2: Remove remaining HTML/XML/SVG Tags
    tag_pattern = re.compile('<.*?>')
    cleaned_text_no_tags = re.sub(tag_pattern, ' ', intermediate_html)

    # 3. STEP 3: Remove Punctuation/Numbers and Lowercase
    cleaned_text_final_str = re.sub(r'[^a-zA-Z\s]', ' ', cleaned_text_no_tags).lower()
    
    # 4. STEP 4: Tokenize and Filter for *Noise*
    potential_words = cleaned_text_final_str.split()
    new_noise_words = []

    for word in potential_words:
        # Rule 1: Filter out words that are too short/long
        if 2 < len(word) < 17:
            
            # Rule 2: Check if it's a known "good" word
            is_english_word = word in ENGLISH_WORDS
            is_ticker_symbol = word in TICKER_SYMBOLS
            
            # Rule 3: Check if it's already in our noise cache
            is_known_noise = word in existing_stopwords
            
            # We want words that are NOT good AND NOT known noise
            if not (is_english_word or is_ticker_symbol) and not is_known_noise:
                new_noise_words.append(word)

    return new_noise_words


def print_frequency_report(title, counter, total_tokens):
    """Helper function to print a formatted top-50 report."""
    print(f"\n{title}")
    print("-" * 40)
    
    if total_tokens == 0:
        print("No new noise tokens found in this batch.")
        print("-" * 40)
        return

    print(f"Total new noise tokens: {total_tokens}")
    top_k = 50
    top_items = counter.most_common(top_k)
    
    print(f"\nTop {top_k} new noise words (candidates to add to stopwords):")
    print(f"{'Rank':<5} {'Word':<20} {'Count':<8} {'Frequency':<10}")
    print(f"{'-'*4:<5} {'-'*19:<20} {'-'*7:<8} {'-'*9:<10}")
    
    for i, (word, count) in enumerate(top_items, 1):
        pct = (count / total_tokens) * 100
        print(f"{i:<5} {word:<20} {count:<8} {pct:>8.2f}%")
    print("-" * 40)


# --- 3. Main Execution ---

if __name__ == "__main__":
    
    parquet_file = "NEWS_20240101-142500_20251101-232422.parquet"

    try:
        df = pd.read_parquet(parquet_file)
    except FileNotFoundError:
        print(f"Error: Parquet file '{parquet_file}' not found.")
        sys.exit(1)
        
    df_sorted = df.sort_values(by='time_published_ts', ascending=True)

    print(f"Loaded {len(df_sorted)} URLs from Parquet.")
    print(df_sorted["url"].head())
    
    # --- Batch Config ---
    BATCH_SIZE = 1000  # Number of *new* items to add per batch
    MAX_WORKERS = 20   # Number of concurrent download threads
    MAX_RETRIES = 2    # Max *retries* (so 3 attempts total: 0, 1, 2)
    
    # Holds (idx, url, retry_count) tuples
    pending_items = [] 
    
    # Aggregate counter for *new noise words*
    aggregate_noise_counter = Counter()
    total_aggregate_noise_tokens = 0

    total_rows = len(df_sorted)
    num_batches = math.ceil(total_rows / BATCH_SIZE)
    print(f"Total rows: {total_rows}. Batch size: {BATCH_SIZE}. Total batches: {num_batches}.")

    for i in range(num_batches):
        print(f"\n=== Starting Batch {i+1}/{num_batches} ===")
        
        # 1. Get new items for this batch
        start_slice_idx = i * BATCH_SIZE
        end_slice_idx = min((i + 1) * BATCH_SIZE, total_rows)
        current_batch_df = df_sorted.iloc[start_slice_idx:end_slice_idx]
        
        # Convert new items to (idx, url, retry_count=0)
        new_items_to_fetch = [(idx, row['url'], 0) for idx, row in current_batch_df.iterrows()]
        
        # Combine new items with any pending (failed) items
        items_to_try = pending_items + new_items_to_fetch
        print(f"Processing {len(items_to_try)} URLs ({len(pending_items)} pending, {len(new_items_to_fetch)} new)")
        
        pending_items = [] # Will be repopulated with this batch's failures
        fetched_results = [] # Will store (idx, url, content)
        
        # --- 2. Fetch all URLs in parallel ---
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_item = {
                executor.submit(fetch_url, idx, url): (idx, url, retry_count) 
                for idx, url, retry_count in items_to_try
            }
            
            for future in as_completed(future_to_item):
                idx, url, retry_count = future_to_item[future]
                _idx, _url, content, error = future.result()
                
                if content is not None:
                    # Success!
                    fetched_results.append((idx, url, content))
                else:
                    # Failure, check retry count
                    if retry_count < MAX_RETRIES:
                        print(f"Failed fetch {idx} (Attempt {retry_count+1}/3): {url}. Retrying next batch.")
                        pending_items.append((idx, url, retry_count + 1))
                    else:
                        print(f"Failed fetch (Attempt 3/3): {url}. Discarding. Error: {error}")

        print(f"Successfully fetched {len(fetched_results)} URLs this batch.")
        print(f"{len(pending_items)} URLs failed, will retry next batch.")

        # --- 3. Process (clean) fetched results sequentially (CPU-bound) ---
        batch_noise_counter = Counter()
        total_batch_noise_tokens = 0
        
        print(f"Cleaning {len(fetched_results)} pages to find new noise words...")
        for idx, url, content in fetched_results:
            try:
                # Find new noise words
                new_noise = extract_potential_noise_words(content, HTML_ATTRIBUTE_STOP_WORDS)
                
                if new_noise:
                    # Update counters for this batch
                    batch_noise_counter.update(new_noise)
                    total_batch_noise_tokens += len(new_noise)
                    
                    # Update aggregate counters
                    aggregate_noise_counter.update(new_noise)
                    total_aggregate_noise_tokens += len(new_noise)
                    
            except Exception as e:
                print(f"Error cleaning {url} (idx {idx}): {e}")
        
        # --- 4. Print Per-Batch Report ---
        print_frequency_report(
            f"--- Batch {i+1} Noise Report ---",
            batch_noise_counter,
            total_batch_noise_tokens
        )
        # --- End of Batch Loop ---

    # --- 5. Final Report (All Batches) ---
    if pending_items:
        print(f"\n--- WARNING ---")
        print(f"{len(pending_items)} URLs failed all attempts and were not processed.")
        # You could save these to a file for review
    
    print_frequency_report(
        "=== Final Aggregate Noise Report (All Batches) ===",
        aggregate_noise_counter,
        total_aggregate_noise_tokens
    )

    print("\nProcess finished.")
