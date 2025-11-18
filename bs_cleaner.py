import os
from bs4 import BeautifulSoup, Comment

HTML_FILE_NAME = "0a0b7d384c34648b.html"
OUTPUT_FILE_NAME = "cleaned_article_text_bs.txt"

def clean_and_extract_article_text(raw_html_content):
    """
    Uses BeautifulSoup to perform a robust cleanup and structured extraction.
    
    1. Parses the HTML structure.
    2. Removes unwanted elements (scripts, styles, comments).
    3. Finds all paragraph (<p>) elements.
    4. Extracts and joins the clean text from these elements.
    """
    
    # Use 'lxml' as the parser for speed and robustness
    soup = BeautifulSoup(raw_html_content, 'lxml')
    
    # --- Step 1: Remove Structured Noise (CSS, JS, Comments) ---
    
    # Remove all <script> tags and their contents
    for script in soup.find_all('script'):
        script.decompose() # .decompose() removes the element and its children from the soup
        
    # Remove all <style> tags and their contents
    for style in soup.find_all('style'):
        style.decompose()
        
    # Remove HTML comments (which often contain metadata or ad placeholders)
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract() # .extract() removes the element but returns it
        
    # Remove common non-article tags (meta, link, header tags) 
    # This is often optional but good for cleaning up header data
    for tag in soup(["meta", "link", "header", "footer", "nav"]):
        tag.decompose()


    # --- Step 2: Extract Desired Content (Paragraphs) ---
    
    # Find all paragraph elements in the remaining, clean HTML structure
    paragraph_elements = soup.find_all('p')
    
    article_content = []
    for p_tag in paragraph_elements:
        # .get_text(strip=True) extracts the text content, removing leading/trailing whitespace
        clean_text = p_tag.get_text(strip=True)
        if clean_text:
            article_content.append(clean_text)
            
    # Join the paragraphs together, separated by two newlines for readability
    return '\n\n'.join(article_content)

# --- Main Execution ---
try:
    # 1. Load the raw HTML content
    with open(HTML_FILE_NAME, 'r', encoding='utf-8') as f:
        raw_html_content = f.read()

    print(f"✅ Successfully loaded {HTML_FILE_NAME}. Size: {len(raw_html_content)} characters.")
    print("")

    # 2. Process the content
    cleaned_article_text = clean_and_extract_article_text(raw_html_content)
    
    print("✨ Successfully cleaned noise and extracted article text using BeautifulSoup.")

    # 3. Save the pure text content to the output file
    with open(OUTPUT_FILE_NAME, 'w', encoding='utf-8') as f:
        f.write(cleaned_article_text)
        
    print(f"💾 Cleaned text saved to {OUTPUT_FILE_NAME}. Size: {len(cleaned_article_text)} characters.")
    
except FileNotFoundError:
    print(f"❌ Error: The file '{HTML_FILE_NAME}' was not found. Please ensure it exists.")
except Exception as e:
    print(f"❌ An unexpected error occurred: {e}")
