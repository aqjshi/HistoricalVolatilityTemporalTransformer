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
import hashlib # NEW: For generating content hash
import secrets # Retained from your partial code
import time 

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
    # Handle the case where the file is missing by starting with an empty set
    HTML_ATTRIBUTE_STOP_WORDS = set() 
    print("Warning: stopwords.json not found. Starting with empty existing noise words set.")
except KeyError:
    print("Error: 'naive_stopwords' key not found in stopwords.json. Starting with empty set.")
    HTML_ATTRIBUTE_STOP_WORDS = set()


# --- 2. Worker Functions ---

def generate_content_hash(content):
    """Generates a short, unique hash based on the content."""
    # Using SHA-256 and returning the first 16 characters for a short, unique ID
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

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
    # ... (Unchanged cleaning and filtering logic) ...
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


def print_frequency_report(title, counter, df_counter, total_documents, total_tokens, output_filename=None):
    """
    Helper function to print a formatted report and calculate/cache TF-IDF scores.
    Uses simplified TF-IDF formula for noise: TF * log(N / DF).
    """
    print(f"\n{title}")
    print("-" * 75)
    
    if total_tokens == 0:
        print("No new noise tokens found.")
        print("-" * 75)
        return

    print(f"Total documents (N): {total_documents}")
    print(f"Total new noise tokens: {total_tokens}")
    top_k = 500
    
    # 1. Calculate TF-IDF Score
    noise_scores = {}
    
    for word, count in counter.items():
        tf = count          # Term Frequency (Count in our corpus of noise words)
        df = df_counter[word] # Document Frequency (Number of documents it appeared in)
        
        # IDF formula: log(Total Documents / Document Frequency)
        # Use log(N / (DF + 1)) to prevent division by zero and smooth the score
        idf = math.log(total_documents / (df + 1)) 
        
        # TF-IDF Score
        score = tf * idf
        noise_scores[word] = score

    # Sort items by TF-IDF score
    top_items = sorted(noise_scores.items(), key=lambda item: item[1], reverse=True)
    top_k_items = top_items[:top_k]

    # 2. Print Report
    print(f"\nTop {top_k} new noise words (ranked by TF-IDF Score):")
    print(f"{'Rank':<5} {'Word':<20} {'Count':<8} {'DF':<5} {'TF-IDF Score':<15} {'Frequency':<10}")
    print(f"{'-'*4:<5} {'-'*19:<20} {'-'*7:<8} {'-'*4:<5} {'-'*14:<15} {'-'*9:<10}")
    
    for i, (word, score) in enumerate(top_k_items, 1):
        count = counter[word]
        df = df_counter[word]
        pct = (count / total_tokens) * 100
        print(f"{i:<5} {word:<20} {count:<8} {df:<5} {score:<15.4f} {pct:>8.2f}%")
    print("-" * 75)

    # 3. Cache Noise Scores to JSON
    if output_filename:
        # Prepare data for caching (Word: TF-IDF Score)
        cache_data = {word: score for word, score in top_items}
        
        try:
            with open(output_filename, 'w') as f:
                json.dump(cache_data, f, indent=4)
            print(f"✅ TF-IDF noise scores (all {len(top_items)} words) cached to: {output_filename}")
        except Exception as e:
            print(f"Error saving noise cache to {output_filename}: {e}")

    # Print the comma-separated list of top words
    formatted_items = [f"\"{name}\"" for name, score in top_k_items]
    print("\nTop words (for easy copy/paste):")
    print(", ".join(formatted_items))
    print("-" * 75)


# --- 3. Main Execution ---

if __name__ == "__main__":
    
    parquet_file = "NEWS_20240101-142500_20251101-232422.parquet"
    NEW_PARQUET_FILE = "NEWS_HASHED_20240101-142500_20251101-232422.parquet" # NEW: Output parquet name
    NOISE_CACHE_FILE = "noise_tfidf_scores.json"
    RAW_TEXT_DIR = "./texts" # NEW: Subdirectory for raw HTML
    
    # 1. Setup Data Structures and Directories
    try:
        df = pd.read_parquet(parquet_file)
    except FileNotFoundError:
        print(f"Error: Parquet file '{parquet_file}' not found.")
        sys.exit(1)
    
    # Prepare DataFrame for saving hashes and track successful downloads
    df['hash_id'] = None 
    df['downloaded'] = False 
    
    # Create the output directory if it doesn't exist
    os.makedirs(RAW_TEXT_DIR, exist_ok=True)
    print(f"Created output directory: {RAW_TEXT_DIR}")
    
    df_sorted = df.sort_values(by='time_published_ts', ascending=True)

    print(f"Loaded {len(df_sorted)} URLs from Parquet.")
    print(df_sorted["url"].head())
    
    # --- Batch Config ---
    BATCH_SIZE = 2000 
    MAX_WORKERS = 20 
    MAX_RETRIES = 2 
    
    # Holds (idx, url, retry_count) tuples
    pending_items = [] 
    
    # Aggregate counters for *new noise words*
    aggregate_noise_counter = Counter() # Holds Term Frequency (TF)
    document_frequency_counter = Counter() # Holds Document Frequency (DF)
    total_aggregate_noise_tokens = 0
    total_processed_documents = 0 # N for IDF calculation

    total_rows = len(df_sorted)
    num_batches = math.ceil(total_rows / BATCH_SIZE)
    print(f"Total rows: {total_rows}. Batch size: {BATCH_SIZE}. Total batches: {num_batches}.")
    
    # Use a lock-free list for thread-safe hash updates (only writing is thread-safe, no reading contention)
    # We will update the main dataframe at the end of the batch
    batch_updates = [] 

    for i in range(num_batches):
        print(f"\n=== Starting Batch {i+1}/{num_batches} ===")
        
        # 1. Get new items for this batch
        start_slice_idx = i * BATCH_SIZE
        end_slice_idx = min((i + 1) * BATCH_SIZE, total_rows)
        current_batch_df = df_sorted.iloc[start_slice_idx:end_slice_idx]
        
        # Convert new items to (idx, url, retry_count=0)
        # Note: We must use .loc to access the original index 'idx' correctly
        new_items_to_fetch = [(idx, row['url'], 0) for idx, row in current_batch_df.iterrows()]
        
        # Combine new items with any pending (failed) items
        items_to_try = pending_items + new_items_to_fetch
        print(f"Processing {len(items_to_try)} URLs ({len(pending_items)} pending, {len(new_items_to_fetch)} new)")
        
        pending_items = [] 
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

        # --- 3. Process, Save, and Update Data Sequentially ---
        
        print(f"Cleaning, saving raw HTML, and updating hash for {len(fetched_results)} pages...")
        for idx, url, content in fetched_results:
            
            # --- NEW STEP: Save Raw HTML and Generate Hash ---
            content_hash = generate_content_hash(content)
            filepath = os.path.join(RAW_TEXT_DIR, f"{content_hash}.txt")
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                # Store update information
                batch_updates.append({'index': idx, 'hash_id': content_hash, 'downloaded': True})
                
            except Exception as e:
                print(f"Error saving raw HTML for {url} (idx {idx}): {e}. Skipping content processing.")
                continue # Skip processing if saving failed

            # --- Noise Word Processing (Retained Logic) ---
            try:
                new_noise = extract_potential_noise_words(content, HTML_ATTRIBUTE_STOP_WORDS)
                
                if new_noise:
                    # Update aggregate counters (TF)
                    aggregate_noise_counter.update(new_noise)
                    total_aggregate_noise_tokens += len(new_noise)

                    # Update document frequency (DF) for this document's unique noise words
                    document_frequency_counter.update(set(new_noise))
                    
                    # Increment total processed documents (N)
                    total_processed_documents += 1
                    
            except Exception as e:
                print(f"Error cleaning {url} (idx {idx}): {e}")
        
        # --- End of Batch Update ---
        # Apply the accumulated hash and download status updates to the main dataframe
        for update in batch_updates:
            df.loc[update['index'], 'hash_id'] = update['hash_id']
            df.loc[update['index'], 'downloaded'] = update['downloaded']

        # Clear batch updates list for the next iteration
        batch_updates = []

    
    # --- Final Step: Save New Parquet File ---
    if pending_items:
        print(f"\n--- WARNING ---")
        print(f"{len(pending_items)} URLs failed all attempts and were not processed.")

    print("\n--- Saving Final Dataframes ---")
    
    # 1. Save the new Parquet file
    df_output = df[df['downloaded'] == True].drop(columns=['downloaded'])
    print(f"Saving new Parquet file with {len(df_output)} rows and 'hash_id' column...")
    df_output.to_parquet(NEW_PARQUET_FILE, index=False)
    print(f"✅ New Parquet file saved as: {NEW_PARQUET_FILE}")

    # 2. Print Final Noise Report and Cache TF-IDF Scores
    print_frequency_report(
        "=== Final Aggregate Noise Report (All Batches - TF-IDF Score) ===",
        aggregate_noise_counter,
        document_frequency_counter,
        total_processed_documents,
        total_aggregate_noise_tokens,
        output_filename=NOISE_CACHE_FILE
    )
