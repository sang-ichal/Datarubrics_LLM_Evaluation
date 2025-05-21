# Use this script when serve_vllm.sh is running

import base64
import urllib.request
import requests
from urllib.parse import urlparse, urlunparse
from tqdm import tqdm
from io import BytesIO
import os
import argparse
import json
import time
import concurrent.futures # Import for ThreadPoolExecutor
from transformers import AutoProcessor

import pandas as pd
import fitz  # PyMuPDF

# Ensure these imports are correct based on your olmocr installation
from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = CUR_DIR
DATA_DIR = os.path.join(ROOT_DIR, "data")
SELECTED_NEURIPS_CSV = os.path.join(DATA_DIR, "csv", "neurips-selected.csv")

# Configuration for VLLM server
VLLM_API_URL = "http://localhost:8100/v1/completions" # Adjust port if different
BATCH_SIZE = 4 # Number of pages to process in one batch request to VLLM
MAX_WORKERS = 8 # Number of concurrent threads to send requests

def forum_to_pdf_url(url):
    """Convert NeurIPS forum URL to PDF URL."""
    parsed_url = urlparse(url)
    path_parts = parsed_url.path.split("/")
    if "forum" in path_parts:
        path_parts[path_parts.index("forum")] = "pdf"
    new_path = "/".join(path_parts)
    return urlunparse(
        (
            parsed_url.scheme,
            parsed_url.netloc,
            new_path,
            parsed_url.params,
            parsed_url.query,
            parsed_url.fragment,
        )
    )
    
def prepare_page_input(processor, file_path, page_num):
    """
    Prepares input for a single page: renders image, builds prompt, applies chat template.
    Returns the prepared text string and original image for reference if needed.
    """
    try:
        # Render page to an image (base64 encoded)
        image_base64 = render_pdf_to_base64png(file_path, page_num, target_longest_image_dim=1500)

        # Build the prompt, using document metadata (anchor text)
        anchor_text = get_anchor_text(file_path, page_num, pdf_engine="pdfreport", target_length=10000)
        prompt = build_finetuning_prompt(anchor_text)

        # Build the full chat messages structure required by the processor
        messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                        ],
                    }
                ]

        # Apply the chat template to get the text input for the model
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # Note: VLLM currently only accepts text input for generation.
        # The image information is encoded into the text_input via the chat template.
        return text_input
    except Exception as e:
        print(f"Error preparing page {page_num} of {file_path}: {e}")
        return None

def send_to_vllm(batch_prompts, original_batch_items):
    """
    Sends a batch of prompts to the VLLM server and returns the generated texts
    along with the original batch items for context.
    """
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "olmOCR-7B-8100", # This should match the --served-model-name from your VLLM script
        "prompt": batch_prompts, # VLLM can take a list of prompts for batching
        "temperature": 0.8,
        "max_tokens": 4096, # Max tokens to generate
        "n": 1, # Number of completions to generate for each prompt
        "stop": ["<|im_end|>", "<|endoftext|>"] # Common stop tokens for chat models
    }

    try:
        response = requests.post(VLLM_API_URL, headers=headers, json=payload, timeout=600) # 10-minute timeout
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        
        results = response.json()
        generated_texts = []
        for choice in results["choices"]:
            generated_texts.append(choice["text"])
        return generated_texts, original_batch_items
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with VLLM server: {e}")
        # Return Nones for failed requests, maintaining structure for processing
        return [None] * len(batch_prompts), original_batch_items

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on PDFs using VLLM server with threading')
    parser.add_argument('--start_file_idx', type=int, default=0,
                        help='Start file index reading.')
    parser.add_argument('--end_file_idx', type=int, default=10,
                        help='End file index reading.')
    args = parser.parse_args()
    
    # Initialize the processor (only needed on the client side for pre-processing)
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Load the CSV containing PDF links and IDs
    df = pd.read_csv(SELECTED_NEURIPS_CSV)
    df = df.sort_values(by='id').iloc[args.start_file_idx : args.end_file_idx]
    url_link_list = df['url_link'].tolist()
    id_list = df['id'].tolist()

    # Use ThreadPoolExecutor for concurrent API requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Store futures for results
        futures = []

        # Process each PDF
        for id_, url_link in tqdm(zip(id_list, url_link_list), desc="Processing PDFs", total=len(id_list)):
            # Ensure output and PDF directories exist
            os.makedirs(os.path.join(DATA_DIR, 'output'), exist_ok=True)
            os.makedirs(os.path.join(DATA_DIR, 'pdf'), exist_ok=True)
            
            # Define paths for PDF and output JSONL
            pdf_path = os.path.join(DATA_DIR, 'pdf', id_)
            save_output_path = os.path.join(DATA_DIR, 'output', f"{id_}.jsonl")
            
            # Download the PDF
            try:
                urllib.request.urlretrieve(forum_to_pdf_url(url_link), pdf_path)
                # print(f"Downloaded PDF: {pdf_path}") # Uncomment for more verbose logging
            except Exception as e:
                print(f"Error downloading PDF {url_link}: {e}")
                continue # Skip to the next PDF if download fails

            # Open the PDF and get the number of pages
            try:
                doc = fitz.open(pdf_path)
                num_pages = len(doc)
                # print(f"Processing {num_pages} pages from {pdf_path}") # Uncomment for more verbose logging
            except Exception as e:
                print(f"Error opening PDF {pdf_path}: {e}")
                continue # Skip to the next PDF if opening fails

            # Collect pages for batch processing
            pages_to_process = []
            for page_num in range(1, num_pages + 1):
                prepared_input = prepare_page_input(processor, pdf_path, page_num)
                if prepared_input:
                    pages_to_process.append({
                        "id": id_,
                        "page_num": page_num,
                        "prompt": prepared_input
                    })

            # Submit batches to the thread pool
            for i in range(0, len(pages_to_process), BATCH_SIZE):
                batch = pages_to_process[i:i + BATCH_SIZE]
                batch_prompts = [item["prompt"] for item in batch]
                # Submit the send_to_vllm function to the executor
                future = executor.submit(send_to_vllm, batch_prompts, batch)
                futures.append(future)

            doc.close() # Close the PDF document after submitting all pages

        # Process results as they complete
        print("\nWaiting for VLLM results to complete...")
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Collecting results"):
            try:
                generated_texts, original_batch_items = future.result()
                for j, text_output in enumerate(generated_texts):
                    original_item = original_batch_items[j]
                    save_output_path = os.path.join(DATA_DIR, 'output', f"{original_item['id']}.jsonl")
                    if text_output is not None:
                        json_obj = {
                            'id': original_item['id'],
                            'page_num': original_item['page_num'],
                            'content': text_output
                        }
                        # Write the output to a JSONL file
                        with open(save_output_path, "a+", encoding='utf-8') as f:
                            f.write(json.dumps(json_obj) + "\n")
                    else:
                        print(f"Skipping writing output for {original_item['id']} page {original_item['page_num']} due to VLLM error.")
            except Exception as exc:
                print(f"Batch generation generated an exception: {exc}")

    print("\nAll processing complete.")
