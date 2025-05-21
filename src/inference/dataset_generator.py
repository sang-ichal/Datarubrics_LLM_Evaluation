import os
import logging
import json
import glob

from .constants import *
from .utils import *
from .templates import *

RUBRIC_SCHEMA_INDICES = {
    0: (SOURCE_TEXT_RUBRIC, os.path.join(SCHEMA_DIR, "source_text.json")),
    1: (SOURCE_IMAGE_RUBRIC, os.path.join(SCHEMA_DIR, "source_image.json")),
    2: (SOURCE_VIDEO_RUBRIC, os.path.join(SCHEMA_DIR, "source_video.json")),
    3: (SOURCE_AUDIO_RUBRIC, os.path.join(SCHEMA_DIR, "source_audio.json")),
    4: (SOURCE_GRAPH_RUBRIC, os.path.join(SCHEMA_DIR, "source_graph.json")),
    5: (SOURCE_TABULAR_RUBRIC, os.path.join(SCHEMA_DIR, "source_tabular.json")),
    6: (SOURCE_TIME_OR_SIGNAL_RUBRIC, os.path.join(SCHEMA_DIR, "source_time_signal.json")),
    7: (ANNOTATIONS_HUMAN_ANNOT_GUIDELINES_RUBRIC, os.path.join(SCHEMA_DIR, "annotations_human_annot_guidelines.json")),
    8: (ANNOTATIONS_MODEL_ANNOT_GUIDELINES_RUBRIC, os.path.join(SCHEMA_DIR, "annotations_model_annot_guidelines.json")),
    9: (ANNOTATIONS_QUALITY_ASSURANCE_RUBRIC, os.path.join(SCHEMA_DIR, "annotations_quality_assurance.json")),
    10: (ANNOTATIONS_DATA_ANNOT_RUBRIC, os.path.join(SCHEMA_DIR, "annotations_data_annot.json")),
    11: (UTILITY_DATA_NOVELTY_RUBRIC, os.path.join(SCHEMA_DIR, "utility_data_novelty.json")),
    12: (UTILITY_TASK_UTILITY_RUBRIC, os.path.join(SCHEMA_DIR, "utility_task_utility.json")),
    13: (UTILITY_HUMAN_LANG_RUBRIC, os.path.join(SCHEMA_DIR, "utility_human_lang.json")),
    14: (UTILITY_NON_HUMAN_LANG_RUBRIC, os.path.join(SCHEMA_DIR, "utility_non_human_lang.json")), 
    15: (UTILITY_DOCUMENTATION_RUBRIC, os.path.join(SCHEMA_DIR, "utility_documentation.json")),
    16: (UTILITY_REPRODUCIBILITY_RUBRIC, os.path.join(SCHEMA_DIR, "utility_reproducibility.json")),
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

def format_data(data):
    return {
        "ID": data["id"],
        "question": data["question"],
        "correct_answers": data["answer"],
        "msg": [{"role": "user", "content": data["prompt"]}],
    }

def get_existing_question_ids(output_path):
        """Read the output file and return a set of existing IDs."""
        existing_ids = set()
        if  os.path.exists(output_path):
            with open(os.path.join(ROOT_DIR, output_path), 'r') as f:
                cur_json = json.load(f)
                for obj in cur_json:
                    existing_ids.add(obj['ID'])
        return existing_ids

def create_dataset(input_folder, output_path, debug=False):
    existing_question_ids = get_existing_question_ids(output_path)
    
    input_list = []
    
    all_jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))
    for jsonl_file in all_jsonl_files:
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
            content_line = page_extract['content'][0]
            try:
                natural_text = json.loads(content_line)['natural_text']
            except Exception as e:
                natural_text = extract_natural_text_simple(content_line)
                
            if natural_text is not None:
                content += " "
                content += natural_text
            
        content = content.strip()

        # Iterate over the rubric and use that content along with {paper_id}-{rubric_index} as the new id
        for rubric_index in range(len(RUBRIC_SCHEMA_INDICES)):
            rubric_pydantic, rubric_schema_path = RUBRIC_SCHEMA_INDICES[rubric_index]
            with open(rubric_schema_path, 'r', encoding='utf-8') as f:
                schema_format = json.load(rubric_schema_path)
            
            if hasattr(rubric_pydantic, 'options'):
                input_list.append({
                    'id': f"{paper_id}-rubric-{rubric_index}",
                    'msg': [{"role": "user", "content": construct_judge_prompt(content,
                                                                               rubric_pydantic,
                                                                               schema_format)}],
                    'schema': schema_format,
                })
            else:
                input_list.append({
                    'id': f"{paper_id}-rubric-{rubric_index}",
                    'msg': [{"role": "user", "content": construct_judge_nooption_prompt(content,
                                                                                        rubric_pydantic,
                                                                                        schema_format)}],
                    'schema': schema_format,
                })
        
    # Process data
    final_dataset_list = []
    for input_item in input_list:
        if input_item['id'] not in existing_question_ids:
            final_dataset_list.append(input_item)
                
    if debug:
        final_dataset_list = final_dataset_list[:DEBUG_COUNT]
                        
    logging.info(f"Final dataset length: {len(final_dataset_list)}")
    
    return final_dataset_list
