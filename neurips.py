import os
import pandas as pd
import requests
import tempfile
from urllib.parse import urlparse, urlunparse
from tqdm import tqdm

from extract import extract_text_from_pdf

def download_pdf(url, temp_file):
    """Download PDF from URL to a temporary file."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(temp_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return temp_file

def forum_to_pdf_url(url):
    """Convert NeurIPS forum URL to PDF URL."""
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.split('/')
    if 'forum' in path_parts:
        path_parts[path_parts.index('forum')] = 'pdf'
    new_path = '/'.join(path_parts)
    return urlunparse((parsed_url.scheme, parsed_url.netloc, new_path, parsed_url.params, parsed_url.query, parsed_url.fragment))

def main():
    os.makedirs('txt', exist_ok=True)
    
    df = pd.read_csv('./csv/NeurIPS.csv')
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing papers"):
        paper_id = row['id']
        forum_url = row['url_link']
        
        pdf_url = forum_to_pdf_url(forum_url)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
            temp_pdf_path = temp_pdf.name
            
        try:
            download_pdf(pdf_url, temp_pdf_path)
            output_path = f"./txt/{paper_id}.txt"
            
            extract_text_from_pdf(
                pdf_path=temp_pdf_path,
                output_path=output_path,
                start_page_i=0,
                end_page_i=None,  # Extract all pages
                extraction_mode="plain"
            )
            
            print(f"Extracted text from {paper_id} to {output_path}")
            
        except Exception as e:
            print(f"Error processing {paper_id}: {e}")
            
        finally:
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)

if __name__ == "__main__":
    main()
