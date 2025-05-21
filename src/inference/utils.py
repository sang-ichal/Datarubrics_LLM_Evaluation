import os
import logging
import json
import re

import numpy as np

from .constants import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def parse_json(output):
    try:
        # Try parsing directly first
        json_out = json.loads(output)
        if not isinstance(json_out, list):
            return [json_out]
        else:
            return json_out
    except json.JSONDecodeError:
        # Clean the output for common issues
        cleaned_output = output.strip()

        # Extract potential JSON objects or arrays
        cleaned_entries = []
        json_object_pattern = re.compile(r'\{.*?}', re.DOTALL)
        entries = json_object_pattern.findall(cleaned_output)
        for entry in entries:
            try:
                # Test if each entry is valid JSON
                json.loads(entry)
                cleaned_entries.append(entry)
            except json.JSONDecodeError:
                # Skip invalid entries
                pass

        # Reconstruct the cleaned JSON array
        cleaned_output = "[" + ",".join(cleaned_entries) + "]"

        # Attempt to parse again
        try:
            return json.loads(cleaned_output)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Error cleaning JSON: {e}")

def write_results(results: list[dict], output_path: str) -> None:
    """Write results as JSON to output path

    Args:
        results (list[dict]): List of dictionary result
        output_path (str): The output path
    """
    # Read existing data from the file (if it exists)
    if os.path.exists(os.path.join(ROOT_DIR, output_path)):
        with open(os.path.join(ROOT_DIR, output_path), 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    with open(os.path.join(ROOT_DIR, output_path), 'w', encoding='utf-8') as f:
        data.extend(results)
        json.dump(data, f, indent=4)
