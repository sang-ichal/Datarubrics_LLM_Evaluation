import os
import logging
import json
import re

# Assuming constants.py is in the same directory or PYTHONPATH is set up
# If constants.py is in the parent directory or a different module structure:
# from ..constants import ROOT_DIR (example if utils.py is in a subdir like 'helpers')
# For simplicity, assuming direct import works based on project structure:
try:
    from .constants import ROOT_DIR
except ImportError:
    # Fallback if running as script or structure differs, assuming ROOT_DIR needs to be defined
    # This might happen if utils.py is not part of a package run with `python -m`
    # A more robust solution depends on your exact project layout and how scripts are called.
    # For now, let's define a fallback ROOT_DIR if constants can't be imported this way.
    # This assumes utils.py is in a subdirectory of the project root.
    CUR_SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))
    ROOT_DIR = os.path.abspath(os.path.join(CUR_SCRIPT_DIR, os.pardir))
    # If constants.py defines ROOT_DIR differently, ensure consistency.
    # print(f"utils.py: ROOT_DIR determined as {ROOT_DIR}")


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def parse_json(output):
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

def write_results(results: list[dict], output_path: str, append: bool = False) -> None:
    """Write results as JSON to output path. Can append or overwrite."""
    # Ensure ROOT_DIR is correctly defined and accessible
    # If ROOT_DIR from constants is not working, this path might be incorrect.
    # This relies on constants.py being correctly imported and ROOT_DIR being set.
    try:
        # Attempt to use ROOT_DIR from constants if available
        from .constants import ROOT_DIR as CONST_ROOT_DIR
        effective_root_dir = CONST_ROOT_DIR
    except ImportError:
        # Fallback if .constants isn't found (e.g. script run directly)
        # This assumes utils.py is in a subdirectory of the project's root.
        current_dir = os.path.dirname(os.path.abspath(__file__))
        effective_root_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
        # Or, if utils.py is at the root of the "package" structure, effective_root_dir = current_dir
        # This needs to be robust based on your project structure.
        # A common pattern:
        # effective_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


    full_output_path = os.path.join(effective_root_dir, output_path)
    
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