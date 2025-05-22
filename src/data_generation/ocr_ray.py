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
import concurrent.futures
from transformers import AutoProcessor
import threading
from collections import Counter

import pandas as pd
import fitz

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = CUR_DIR
DATA_DIR = os.path.join(ROOT_DIR, "data")
SELECTED_CSV = os.path.join(DATA_DIR, "csv", "filtered_year_iclr_conference.csv")
PDF_DIR = os.path.join(DATA_DIR, "pdf")
SAVE_DIR_NAME = os.path.basename(SELECTED_CSV).replace(".csv", "")

# Global variables for round-robin distribution of VLLM URLs
_vllm_url_index = 0
_vllm_url_lock = threading.Lock()
_file_locks = {} 
_log_file_lock = threading.Lock() 

PORTS = [8100, 8200, 8300]
VLLM_API_URLS = [
    f"http://localhost:{port}/v1/completions"
    for port in PORTS
]

BATCH_SIZE = 4 # Number of pages per VLLM call / per worker task
MAX_WORKERS = 64 # Number of concurrent worker threads

def get_file_lock(pdf_id):
    if pdf_id not in _file_locks:
        _file_locks[pdf_id] = threading.Lock()
    return _file_locks[pdf_id]

def forum_to_pdf_url(url):
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
    try:
        image_base64 = render_pdf_to_base64png(file_path, page_num, target_longest_image_dim=1500)
        anchor_text = get_anchor_text(file_path, page_num, pdf_engine="pdfreport", target_length=10000)
        prompt = build_finetuning_prompt(anchor_text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]
        text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text_input
    except Exception as e:
        print(f"Error preparing page {page_num} of {file_path}: {e}")
        return None

def send_to_vllm(batch_prompts, original_batch_items):
    global _vllm_url_index, _vllm_url_lock

    with _vllm_url_lock:
        vllm_api_url = VLLM_API_URLS[_vllm_url_index]
        _vllm_url_index = (_vllm_url_index + 1) % len(VLLM_API_URLS)
    
    port = vllm_api_url.split(":")[2].split("/")[0]
    model_name = f"olmOCR-7B-{port}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "prompt": batch_prompts,
        "temperature": 0.8,
        "max_tokens": 4096,
        "n": 1,
        "stop": ["<|im_end|>", "<|endoftext|>"]
    }

    try:
        response = requests.post(vllm_api_url, headers=headers, json=payload, timeout=600)
        response.raise_for_status()
        results = response.json()
        generated_texts = [choice["text"] for choice in results["choices"]]
        return generated_texts, original_batch_items
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with VLLM server ({vllm_api_url}): {e}")
        return [None] * len(batch_prompts), original_batch_items

def write_results_to_jsonl(original_item, text_output, output_dir):
    save_output_path = os.path.join(output_dir, f"{original_item['id']}.jsonl")
    json_obj = {
        'id': original_item['id'],
        'page_num': original_item['page_num'],
        'content': text_output
    }
    with get_file_lock(original_item['id']):
        with open(save_output_path, "a+", encoding='utf-8') as f:
            f.write(json.dumps(json_obj) + "\n")

def mark_pdf_as_fully_processed(pdf_id, processed_files_log):
    with _log_file_lock: 
        with open(processed_files_log, "a") as f:
            f.write(f"{pdf_id}\n")

def get_fully_processed_pdfs(processed_files_log):
    if not os.path.exists(processed_files_log):
        return set()
    with _log_file_lock: 
        with open(processed_files_log, "r") as f:
            return set(line.strip() for line in f)

def get_pages_processed_from_jsonl(pdf_id, output_dir):
    jsonl_path = os.path.join(output_dir, f"{pdf_id}.jsonl")
    processed_pages = set()
    with get_file_lock(pdf_id):
        if os.path.exists(jsonl_path):
            try:
                with open(jsonl_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            if 'page_num' in data:
                                processed_pages.add(data['page_num'])
                        except json.JSONDecodeError:
                            print(f"Warning: Malformed JSON line in {jsonl_path}: {line.strip()}")
            except Exception as e:
                print(f"Error reading existing JSONL for {pdf_id}: {e}")
    return processed_pages

def prepare_and_send_batch_worker(processor_obj, micro_batch_descriptors):
    """
    Worker function: prepares inputs for a micro-batch of pages and sends them to VLLM.
    Returns (generated_texts, list_of_original_item_details_for_successful_sends)
    """
    batch_prompts = []
    # These are the original items corresponding to successfully prepared prompts
    prepared_original_items = [] 

    for task_desc in micro_batch_descriptors:
        pdf_id = task_desc['id']
        page_num = task_desc['page_num']
        pdf_path = task_desc['pdf_path']
        
        prepared_input = prepare_page_input(processor_obj, pdf_path, page_num)
        if prepared_input:
            batch_prompts.append(prepared_input)
            prepared_original_items.append(task_desc) # task_desc already has id, page_num, total_pdf_pages
        else:
            print(f"Warning: Could not prepare page {page_num} for {pdf_id} in worker. Skipping in this batch.")

    if not batch_prompts:
        return [], [] # No successfully prepared prompts in this micro-batch

    generated_texts, items_sent_to_vllm = send_to_vllm(batch_prompts, prepared_original_items)
    return generated_texts, items_sent_to_vllm


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on PDFs using VLLM server with on-the-fly preprocessing by worker threads.')
    parser.add_argument('--start_file_idx', type=int, default=0, help='Start file index reading.')
    parser.add_argument('--end_file_idx', type=int, default=3000, help='End file index reading.')
    args = parser.parse_args()
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    output_dir = os.path.join(DATA_DIR, 'output', SAVE_DIR_NAME)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output will be saved to: {output_dir}")

    processed_pdfs_log = os.path.join(DATA_DIR, 'output', f"{SAVE_DIR_NAME}_processed_pdfs.log")
    fully_processed_ids = get_fully_processed_pdfs(processed_pdfs_log)
    print(f"Found {len(fully_processed_ids)} PDFs already fully processed.")

    df = pd.read_csv(SELECTED_CSV)
    df = df.sort_values(by='id').iloc[args.start_file_idx : args.end_file_idx]
    id_list = df['id'].tolist()

    # --- Phase 1: Identify raw tasks (pages to process) ---
    print("\n--- Phase 1: Identifying raw tasks for processing ---")
    raw_page_tasks = [] 

    for id_ in tqdm(id_list, desc="Scanning PDFs for pages to process"):
        if id_ in fully_processed_ids:
            continue 

        pdf_path = os.path.join(PDF_DIR, SAVE_DIR_NAME, id_)
        if not os.path.exists(pdf_path):
            print(f"Skipping {id_}: PDF not found at {pdf_path}. Marking as fully processed.")
            if id_ not in fully_processed_ids: mark_pdf_as_fully_processed(id_, processed_pdfs_log)
            fully_processed_ids.add(id_)
            continue

        try:
            doc = fitz.open(pdf_path)
            num_pages = len(doc)
        except Exception as e:
            print(f"Error opening PDF {pdf_path}: {e}. Marking as fully processed.")
            if id_ not in fully_processed_ids: mark_pdf_as_fully_processed(id_, processed_pdfs_log)
            fully_processed_ids.add(id_)
            continue

        pages_already_saved_in_jsonl = get_pages_processed_from_jsonl(id_, output_dir)
        
        if num_pages > 0 and len(pages_already_saved_in_jsonl) == num_pages:
            if id_ not in fully_processed_ids:
                print(f"PDF {id_} already has all {num_pages} pages saved. Marking as fully processed.")
                mark_pdf_as_fully_processed(id_, processed_pdfs_log)
                fully_processed_ids.add(id_)
            doc.close()
            continue
        
        if num_pages == 0:
            print(f"PDF {id_} has 0 pages. Marking as fully processed.")
            if id_ not in fully_processed_ids: mark_pdf_as_fully_processed(id_, processed_pdfs_log)
            fully_processed_ids.add(id_)
            doc.close()
            continue

        for page_num in range(1, num_pages + 1):
            if page_num not in pages_already_saved_in_jsonl:
                raw_page_tasks.append({
                    "id": id_,
                    "page_num": page_num,
                    "pdf_path": pdf_path, # Pass the PDF path for preparation
                    "total_pdf_pages": num_pages 
                })
        doc.close()

    print(f"Identified {len(raw_page_tasks)} raw page tasks for processing.")
    if not raw_page_tasks:
        print("No tasks remaining to process. Exiting.")
        exit() 

    # --- Initialize PDF Progress Tracker ---
    pdf_progress_tracker = {} 
    # Using a set to track which PDFs are part of the current run for accurate tracker initialization
    pdfs_in_current_raw_tasks = set()
    for task_desc in raw_page_tasks:
        pdfs_in_current_raw_tasks.add(task_desc['id'])

    for pdf_id in pdfs_in_current_raw_tasks:
        # For each unique PDF ID in the tasks, get its total pages and current saved count
        # This assumes total_pdf_pages is consistent for all tasks of the same PDF_id,
        # which it should be as derived when iterating `id_list`.
        # A more robust way is to re-fetch from the first task_desc for this pdf_id or re-open PDF.
        # For simplicity, we find the first task for this PDF to get its total_pdf_pages.
        
        first_task_for_pdf = next((task for task in raw_page_tasks if task['id'] == pdf_id), None)
        total_pages_for_this_pdf = 0
        if first_task_for_pdf:
            total_pages_for_this_pdf = first_task_for_pdf['total_pdf_pages']
        else: # Should not happen if pdf_id came from raw_page_tasks
            try: # Fallback: try to open PDF again to get page count
                temp_doc = fitz.open(os.path.join(PDF_DIR, SAVE_DIR_NAME, pdf_id))
                total_pages_for_this_pdf = len(temp_doc)
                temp_doc.close()
            except Exception as e:
                print(f"Could not determine total pages for {pdf_id} for progress tracker: {e}")


        saved_pages_count = len(get_pages_processed_from_jsonl(pdf_id, output_dir))
        pdf_progress_tracker[pdf_id] = {
            'total_pages': total_pages_for_this_pdf,
            'pages_completed_count': saved_pages_count 
        }
        
    # --- Phase 2: Process tasks using ThreadPoolExecutor ---
    print("\n--- Phase 2: Submitting tasks for preprocessing and VLLM ---")
    futures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for i in tqdm(range(0, len(raw_page_tasks), BATCH_SIZE), desc="Submitting micro-batches to workers"):
            micro_batch_descriptors = raw_page_tasks[i:i + BATCH_SIZE]
            futures.append(executor.submit(prepare_and_send_batch_worker, processor, micro_batch_descriptors))

        # Collect results and update progress
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Collecting VLLM results"):
            try:
                generated_texts, original_items_from_batch = future.result()
                for j, text_output in enumerate(generated_texts):
                    original_item_detail = original_items_from_batch[j]
                    pdf_id = original_item_detail['id']
                    page_num = original_item_detail['page_num']
                    
                    if text_output is not None:
                        write_results_to_jsonl(original_item_detail, text_output, output_dir)
                        if pdf_id in pdf_progress_tracker:
                            pdf_progress_tracker[pdf_id]['pages_completed_count'] += 1
                        else:
                            # This might happen if a PDF ID was in tasks but couldn't be initialized in tracker
                            print(f"Warning: {pdf_id} not found in progress tracker. Page {page_num} processed.")
                    else:
                        print(f"Skipping writing output for {pdf_id} page {page_num} due to VLLM error in batch.")
            except Exception as exc:
                # This catches errors from future.result() itself or unhandled exceptions in worker
                print(f"A batch processing task generated an exception: {exc}")


    # --- Finalization: Check and mark fully processed PDFs ---
    print("\n--- Finalization: Marking fully processed PDFs ---")
    final_incomplete_tasks_count = 0

    # Re-evaluate all PDFs that were part of this run's tasks or previously in tracker
    all_relevant_pdfs_for_final_check = set(pdf_progress_tracker.keys())
    # Add any PDFs that might have only had failed preparations and thus not in tracker updates
    for task_desc in raw_page_tasks:
        all_relevant_pdfs_for_final_check.add(task_desc['id'])


    for pdf_id in all_relevant_pdfs_for_final_check:
        info = pdf_progress_tracker.get(pdf_id)
        if not info: # If PDF was in raw_tasks but not tracker (e.g. all preps failed or init issue)
            try:
                doc = fitz.open(os.path.join(PDF_DIR, SAVE_DIR_NAME, pdf_id))
                total_pages = len(doc)
                doc.close()
            except Exception: total_pages = 0
            saved_pages_count = len(get_pages_processed_from_jsonl(pdf_id, output_dir))
            info = {'total_pages': total_pages, 'pages_completed_count': saved_pages_count}
            pdf_progress_tracker[pdf_id] = info # Add to tracker for consistent logic below
        else: # Refresh saved pages count for those already in tracker
            info['pages_completed_count'] = len(get_pages_processed_from_jsonl(pdf_id, output_dir))


        if info['total_pages'] > 0 and info['pages_completed_count'] == info['total_pages']:
            if pdf_id not in fully_processed_ids:
                mark_pdf_as_fully_processed(pdf_id, processed_pdfs_log)
                print(f"PDF {pdf_id}: All {info['total_pages']} pages processed and saved. Marked complete.")
                fully_processed_ids.add(pdf_id) # Update current session's set
        else:
            if info['total_pages'] > 0:
                pages_still_pending = info['total_pages'] - info['pages_completed_count']
                if pages_still_pending > 0:
                    print(f"PDF {pdf_id}: Incomplete. {info['pages_completed_count']}/{info['total_pages']} pages saved. {pages_still_pending} pages remain.")
                    final_incomplete_tasks_count += pages_still_pending
            elif info['total_pages'] == 0 and pdf_id not in fully_processed_ids:
                 print(f"PDF {pdf_id}: Problematic (e.g., 0 pages or read error). Saved: {info['pages_completed_count']}. Consider marking complete if no work possible.")


    if final_incomplete_tasks_count == 0:
        print("All identified tasks appear to be processed, or corresponding PDFs are now complete.")
    else:
        print(f"A total of {final_incomplete_tasks_count} page tasks may remain incomplete. Rerun script to re-attempt.")

    print("\nAll script execution complete.")