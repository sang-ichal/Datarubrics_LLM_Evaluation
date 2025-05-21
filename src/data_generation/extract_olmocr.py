import base64
import urllib.request
import requests
from urllib.parse import urlparse, urlunparse
from tqdm import tqdm
from io import BytesIO
import os
import argparse
import json

from PIL import Image
import torch
import pandas as pd
import fitz  # PyMuPDF
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from olmocr.data.renderpdf import render_pdf_to_base64png
from olmocr.prompts import build_finetuning_prompt
from olmocr.prompts.anchor import get_anchor_text

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(CUR_DIR)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SELECTED_NEURIPS_CSV = os.path.join(DATA_DIR, "csv", "neurips-selected.csv")

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
    
def process_page(model, processor, file_path, page_num):
    # Render page 1 to an image
    image_base64 = render_pdf_to_base64png(file_path, page_num, target_longest_image_dim=1500)

    # Build the prompt, using document metadata
    anchor_text = get_anchor_text(file_path, page_num, pdf_engine="pdfreport", target_length=10000)
    prompt = build_finetuning_prompt(anchor_text)

    # Build the full prompt
    messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                    ],
                }
            ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    main_image = Image.open(BytesIO(base64.b64decode(image_base64)))

    inputs = processor(
        text=[text],
        images=[main_image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(next(model.parameters()).device) for (key, value) in inputs.items()}

    # Generate the output
    output = model.generate(
                **inputs,
                temperature=0.8,
                max_new_tokens=4096,
                num_return_sequences=1,
                do_sample=True,
            )

    # Decode the output
    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = output[:, prompt_length:]
    text_output = processor.tokenizer.batch_decode(
        new_tokens, skip_special_tokens=True
    )
    
    return text_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on PDFs')
    parser.add_argument('--start_file_idx', type=int, default=0,
                        help='Start file index reading.')
    parser.add_argument('--end_file_idx', type=int, default=101,
                        help='End file index reading.')
    parser.set_defaults(debug=False, rewrite_output=False)
    args = parser.parse_args()
    
    # Initialize model
    model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview",
                                                            torch_dtype=torch.bfloat16,
                                                            device_map="auto").eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    df = pd.read_csv(SELECTED_NEURIPS_CSV)
    df = df.sort_values(by='id').iloc[args.start_file_idx : args.end_file_idx]
    url_link_list = df['url_link'].tolist()
    id_list = df['id'].tolist()

    for id_, url_link in tqdm(zip(id_list, url_link_list)):
        # Ensure directory exists
        os.makedirs(os.path.join(DATA_DIR, 'output'), exist_ok=True)
        os.makedirs(os.path.join(DATA_DIR, 'pdf'), exist_ok=True)
        
        # Grab PDF
        pdf_path = os.path.join(DATA_DIR, 'pdf', id_)
        save_output_path = os.path.join(DATA_DIR, 'output', f"{id_}.jsonl")
        urllib.request.urlretrieve(forum_to_pdf_url(url_link), pdf_path)

        # Check length of document
        doc = fitz.open(pdf_path)
        num_pages = len(doc)

        for page_num in tqdm(range(1, num_pages + 1)):
            text_output = process_page(model, processor, pdf_path, page_num)
            json_obj = {
                'id': id_,
                'page_num': page_num,
                'content': text_output
            }

            with open(save_output_path, "a+", encoding='utf-8') as f:
                f.write(json.dumps(json_obj) + "\n")
