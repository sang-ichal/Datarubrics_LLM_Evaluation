import os
import logging
import json
import glob
from tqdm import tqdm

from .constants import *
from .utils import *
from .templates import *

RUBRIC_SCHEMA_INDICES = {
    0: (None, os.path.join(SCHEMA_DIR, "sources.json")),
    1: (None, os.path.join(SCHEMA_DIR, "annotations_data_annotations.json")),
    2: (ANNOTATIONS_QUALITY_ASSURANCE_RUBRIC, os.path.join(SCHEMA_DIR, "annotations_quality_assurance.json")),
    3: (UTILITY_DATA_NOVELTY_RUBRIC, os.path.join(SCHEMA_DIR, "utility_data_novelty.json")),
    4: (UTILITY_TASK_UTILITY_RUBRIC, os.path.join(SCHEMA_DIR, "utility_task_utility.json")),
    5: (UTILITY_LANG_RUBRIC, os.path.join(SCHEMA_DIR, "utility_lang.json")),
    6: (UTILITY_REPRODUCIBILITY_RUBRIC, os.path.join(SCHEMA_DIR, "utility_reproducibility.json")),
}

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

def get_existing_question_ids(output_path):
        """Read the output file and return a set of existing IDs."""
        existing_ids = set()
        if os.path.exists(output_path):
            with open(os.path.join(ROOT_DIR, output_path), 'r') as f:
                cur_json = json.load(f)
                for obj in cur_json:
                    existing_ids.add(obj['id'])
        return existing_ids

def create_dataset(input_folder, output_path,
                   start_idx=0, end_idx=-1,
                   combine_prompts=False, debug=False):
    existing_question_ids = get_existing_question_ids(output_path)
    
    input_list = []
    
    all_jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    
    logging.info("Reading over the files!")
    for jsonl_file in tqdm(all_jsonl_files):
        content_list = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                content_list.append(json.loads(line))
            
        # Sort based on 'page_num' field
        content_list = sorted(content_list, key=lambda x: x.get('page_num'))
        paper_id = content_list[0]['id']
        
        # Combine content of the paper into one
        content = ""
        for page_extract in content_list:
            if isinstance(page_extract['content'], list):
                content_line = page_extract['content'][0]
            elif isinstance(page_extract['content'], str):
                content_line = page_extract['content']
            else:
                raise ValueError("Seems like `page_extract['content']` is neither string or list, unexpected error!")
            try:
                natural_text = json.loads(content_line)['natural_text']
            except Exception as e:
                natural_text = extract_natural_text_simple(content_line)
                
            if natural_text is not None:
                content += " "
                content += natural_text
            
        content = content.strip()

        if combine_prompts:
            # Combine all rubrics in one prompt
            rubric_with_options = []
            rubric_no_options = []
            schema_dicts = {}
            for rubric_index in range(len(RUBRIC_SCHEMA_INDICES)):
                rubric_pydantic, rubric_schema_path = RUBRIC_SCHEMA_INDICES[rubric_index]
                with open(rubric_schema_path, 'r', encoding='utf-8') as f:
                    schema_format = json.load(f)
                    
                schema_name = schema_format['name']
                if len(rubric_pydantic.options) != 0:
                    rubric_with_options.append(rubric_pydantic)
                    key_name = schema_format['schema']['required']
                    schema_dicts[schema_name] = schema_format['schema']['properties'][key_name]
                else:
                    rubric_no_options.append(rubric_pydantic)
                    schema_dicts[schema_name] = schema_format['schema']['properties']
            
            combined_schema = {
                "name": "all_rubrics",
                "schema": {
                    "type": "object",
                    "properties": schema_dicts,
                    "required": list(schema_dicts.keys()),
                    "additionalProperties": False  
                },
                "strict": True
            }
            prompt = construct_judge_combine_all_prompt(content, rubric_with_options,
                                                        rubric_no_options, json.dumps(combined_schema, indent=2))
            input_list.append({
                    'id': f"{paper_id}",
                    'msg': [{"role": "user", "content": prompt}],
                    'schema': combined_schema['schema'],
                })
        else:
            rubric_pydantic, rubric_schema_path = RUBRIC_SCHEMA_INDICES[0]
            with open(rubric_schema_path, 'r', encoding='utf-8') as f:
                schema_format = json.load(f)
            input_list.append({
                'id': f"{paper_id}-rubric-0",
                'msg': [{"role": "user", "content": construct_sources_prompt(content,
                                                                            json.dumps(schema_format['schema'], indent=2))}],
                'schema': schema_format,
            })
            
            rubric_pydantic, rubric_schema_path = RUBRIC_SCHEMA_INDICES[1]
            with open(rubric_schema_path, 'r', encoding='utf-8') as f:
                schema_format = json.load(f)
            input_list.append({
                'id': f"{paper_id}-rubric-1",
                'msg': [{"role": "user", "content": construct_data_annotations_prompt(content,
                                                                            json.dumps(schema_format['schema'], indent=2))}],
                'schema': schema_format,
            })
            
            # Iterate over the rubric and use that content along with {paper_id}-{rubric_index} as the new id
            for rubric_index in range(2, len(RUBRIC_SCHEMA_INDICES)):
                rubric_pydantic, rubric_schema_path = RUBRIC_SCHEMA_INDICES[rubric_index]
                with open(rubric_schema_path, 'r', encoding='utf-8') as f:
                    schema_format = json.load(f)
                
                if len(rubric_pydantic.options) != 0:
                    input_list.append({
                        'id': f"{paper_id}-rubric-{rubric_index}",
                        'msg': [{"role": "user", "content": construct_judge_prompt(content,
                                                                                rubric_pydantic,
                                                                                json.dumps(schema_format['schema'], indent=2))}],
                        'schema': schema_format,
                    })
                else:
                    input_list.append({
                        'id': f"{paper_id}-rubric-{rubric_index}",
                        'msg': [{"role": "user", "content": construct_judge_nooption_prompt(content,
                                                                                            rubric_pydantic,
                                                                                            json.dumps(schema_format['schema'], indent=2))}],
                        'schema': schema_format,
                    })
        
    # Process data
    input_list = sorted(input_list, key=lambda x: x.get('id'))
    if end_idx < 0:
        input_list = input_list[start_idx:]
    else:
        input_list = input_list[start_idx:end_idx]
        
    final_dataset_list = []
    for input_item in input_list:
        if input_item['id'] not in existing_question_ids:
            final_dataset_list.append(input_item)
                
    if debug:
        final_dataset_list = final_dataset_list[:DEBUG_COUNT]
                        
    logging.info(f"Final dataset length: {len(final_dataset_list)}")
    
    return final_dataset_list
