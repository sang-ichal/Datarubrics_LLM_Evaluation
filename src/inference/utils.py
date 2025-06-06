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

def ray_parse_json(output):
    if output is None:
        return None
    try:
        json_out = json.loads(output)
        return json_out
    except json.JSONDecodeError:
        cleaned_output = str(output).strip()
        
        match = re.search(r'```json\s*(.*?)\s*```', cleaned_output, re.DOTALL)
        if match:
            cleaned_output = match.group(1)
        else:
            start_brace = cleaned_output.find('{')
            start_bracket = cleaned_output.find('[')
            first_char_index = -1
            
            if start_brace != -1 and start_bracket != -1:
                first_char_index = min(start_brace, start_bracket)
            elif start_brace != -1:
                first_char_index = start_brace
            elif start_bracket != -1:
                first_char_index = start_bracket
            
            if first_char_index != -1:
                if cleaned_output[first_char_index] == '{':
                    last_char_index = cleaned_output.rfind('}')
                else:
                    last_char_index = cleaned_output.rfind(']')
                
                if last_char_index > first_char_index:
                    cleaned_output = cleaned_output[first_char_index : last_char_index+1]

        try:
            return json.loads(cleaned_output)
        except json.JSONDecodeError as e:
            logging.error(f"Failed to parse JSON after cleaning. Error: {e}. Output snippet: {output[:500]}...")
            return {"error": "JSONDecodeError", "message": str(e), "original_output": output}

def ray_write_results(results: list[dict], output_path: str, append: bool = False) -> None:
    """Write results as JSON to output path. Can append or overwrite."""
    full_output_path = os.path.join(ROOT_DIR, output_path)
    
    data_to_write = []
    if append and os.path.exists(full_output_path):
        try:
            with open(full_output_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                if isinstance(existing_data, list):
                    data_to_write.extend(existing_data)
                else:
                    logging.warning(f"Existing data in {output_path} is not a list. Overwriting.")
        except json.JSONDecodeError:
            logging.warning(f"Could not decode existing JSON from {output_path}. Overwriting.")
        except Exception as e:
            logging.error(f"Error reading existing data from {output_path}: {e}. Overwriting.")

    if isinstance(results, list):
        data_to_write.extend(results)
    else:
        data_to_write.append(results) # Should ideally always be a list from completion functions

    os.makedirs(os.path.dirname(full_output_path), exist_ok=True)
    
    try:
        with open(full_output_path, 'w', encoding='utf-8') as f:
            json.dump(data_to_write, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to write results to {full_output_path}: {e}")
