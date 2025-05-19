import glob 
import json
import os
import argparse
import logging

def extract_natural_text_simple(line):
    key = '"natural_text":'
    idx = line.find(key)
    if idx == -1:
        return None
    after = line[idx + len(key):].lstrip()

    # Handle if starts with quote (most likely case)
    if after.startswith('"'):
        after = after[1:]  # skip the starting quote

        # Now find the ending quote safely
        end = len(after)
        escape = False
        for i, c in enumerate(after):
            if c == '"' and not escape:
                end = i
                break
            escape = (c == '\\') and not escape

        text = after[:end]
        try:
            return bytes(text, "utf-8").decode("raw_unicode_escape")
        except Exception:
            return text
    else:
        return after.rstrip("}")  # fallback if no quote


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert OCR jsonl(s) to txt')
    parser.add_argument('--input_folder', type=str,
                        help='Input folder containing *.jsonl.')
    parser.add_argument('--output_folder', type=str,
                        help='Output folder to put *.txt.')
    parser.set_defaults(debug=False, rewrite_output=False)
    args = parser.parse_args()
    
    list_output = glob.glob(os.path.join(args.input_folder, "*.jsonl"))

    for o_f in list_output:
        id_ = None
        content = ""
        try:
            with open(o_f, 'r') as f:
                for i, l in enumerate(f.readlines()):
                    line = json.loads(l)
                    content_line = line['content'][0]
                    try:
                        natural_text = json.loads(content_line)['natural_text']
                    except Exception as e:
                        natural_text = extract_natural_text_simple(content_line)
                        
                    if natural_text is not None:
                        content += " "
                        content += natural_text
                        id_ = line['id']
            
            content = content.strip()
            
            with open(os.path.join(args.output_folder, f"{id_}.txt"), 'w') as f:
                f.write(content)
        except Exception as e:
            logging.warning(f"Error for id `{id_}` at {i + 1} with error {e}")
