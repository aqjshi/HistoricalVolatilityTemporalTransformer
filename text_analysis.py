import json 
import nltk
import os
from nltk.corpus import words
from collections import Counter
import sys

INPUT_FILENAME = "noise_tfidf_scores_compressed.json"
RELEVANT_CANDIDATES_FILE = "set_to_keep.json"
CLEAN_NOISE_FILE = "set_to_remove.json"

try:
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading 'words' corpus for dictionary check...")
    nltk.download('words')

ENGLISH_WORDS = set(words.words())
TICKER_SYMBOLS = {"meta", "amd", "aapl", "msft", "tsla", "nvda"}


# --- 2. Filter Function ---
def is_likely_relevant(word, score):
    vowels = 'aeiou'
    vowel_count = sum(1 for char in word if char in vowels)
    
    vowel_ratio = vowel_count / len(word) if len(word) > 0 else 0
    if len(word) >= 6 and vowel_ratio >= 0.35:
        return True
    
    return False
def is_valid_english_or_ticker(word, score):
    """
    Checks if a word is in the NLTK English dictionary or a predefined 
    set of financial ticker symbols.
    """
    # Convert word to lowercase for comparison with sets
    word_lower = word.lower()
    
    # Check if the word is in either the English dictionary or the ticker list
    is_english = word_lower in ENGLISH_WORDS
    is_ticker = word_lower in TICKER_SYMBOLS
    
    return is_english or is_ticker

def separate_tfidf(input_file, relevant_file, noise_file, filter_func):
    with open(input_file, 'r') as f:
        data = json.load(f)

    set_to_keep, set_to_remove = {}, {} 

    for word, score in data.items():

        if filter_func(word, score): 
            set_to_keep[word] = score 
        else:
            set_to_remove[word] = score


    with open(relevant_file, 'w') as f:
        json.dump(set_to_keep, f, indent=4)


    with open(noise_file, 'w') as f:
        json.dump(set_to_remove, f, indent=4)



if __name__ == "__main__":
    separate_tfidf(INPUT_FILENAME, RELEVANT_CANDIDATES_FILE, CLEAN_NOISE_FILE, is_valid_english_or_ticker)
