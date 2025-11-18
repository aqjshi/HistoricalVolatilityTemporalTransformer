import re
import os


HTML_FILE_NAME = "0a0b7d384c34648b.html"
OUTPUT_FILE_NAME = "cleaned_text.txt"

def remove_common_noise_elements(raw_html_content):
    """
    Removes <script>, <style>, and <meta> blocks from HTML content.
    
    <meta> tags are typically self-closing, so they are handled differently
    than <script> and <style> which contain body content.
    
    Args:
        raw_html_content (str): The full HTML content as a string.
        
    Returns:
        str: The HTML content with noise elements removed.
    """
    
    # Flag to enable multi-line matching (DOTALL) and case-insensitive matching (IGNORECASE)
    flags = re.IGNORECASE | re.DOTALL
    
    processed_content = raw_html_content

    # 1. Remove JavaScript blocks (<script>...</script>)
    # Matches the opening tag, content, and closing tag non-greedily.
    processed_content = re.sub(
        r'<script\b[^>]*>[\s\S]*?</script>', 
        '', 
        processed_content, 
        flags=flags
    )

    # 2. Remove CSS blocks (<style>...</style>)
    processed_content = re.sub(
        r'<style\b[^>]*>[\s\S]*?</style>', 
        '', 
        processed_content, 
        flags=flags
    )


    processed_content = re.sub(
        r'<meta\b[^>]*>', 
        '', 
        processed_content, 
        flags=flags
    )

    # 4. Remove Link tags (<link ... >) - often used for CSS/Favicons/RSS
    processed_content = re.sub(
        r'<link\b[^>]*>', 
        '', 
        processed_content, 
        flags=flags
    )
    
    # 5. Remove Comment tags (<!-- ... -->) - often contain hidden noise
    processed_content = re.sub(
        r'<!--[\s\S]*?-->', 
        '', 
        processed_content, 
        flags=flags
    )
    processed_content = re.sub(
        r'<nav\b[^>]*>', 
        '', 
        processed_content, 
        flags=flags
    )

    return processed_content


with open(HTML_FILE_NAME, 'r', encoding='utf-8') as f:
    raw_html_content = f.read()


cleaned_content = remove_common_noise_elements(raw_html_content)


with open(OUTPUT_FILE_NAME, 'w', encoding='utf-8') as f:
    f.write(cleaned_content)
