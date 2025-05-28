import os
import logging
import argparse
import json

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch

CUR_DIR = os.path.abspath(os.path.dirname(__file__))

# Define for local models so only one time initialization
MODEL = None
TOKENIZER = None
DEBUG_COUNT = 10

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

INSTRUCTION_FORMAT = """
    ### TASK
    Given the title and abstract of a paper, determine whether the paper introduces a new dataset. Respond with "true" if the paper introduces a new dataset; otherwise, respond with "false".

    ### INPUT
    Title:
    {title}

    Abstract:
    {abstract}

    ### OUTPUT FORMAT
    Return a JSON response in the following format:
    {{
        "explanation": "Very short explanation why the answer is true or false",
        "score": "Final boolean answer between true or false"
    }}

    ### RESPONSE
    """

def vllm_completion(messages, dataset_ids, output_file_name):
    list_text = TOKENIZER.apply_chat_template(messages,
                                            tokenize=False,
                                            add_generation_prompt=True)
    
    outputs = MODEL.generate(list_text, SAMPLING_PARAMS)

    # Print the outputs.
    result_list = []
    for output, id_ in zip(outputs, dataset_ids):
        result_list.append({
            'id': id_,
            'response': output.outputs[0].text
        })
        
    # Read existing data from the file (if it exists)
    if os.path.exists(output_file_name):
        with open(output_file_name, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    with open(output_file_name, 'w', encoding='utf-8') as f:
        data.extend(result_list)
        json.dump(data, f, indent=4)    

# prepare the model input
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check whether the paper introduce new benchmark dataset')
    parser.add_argument('--model_config_path', '-c', type=str, required=True,
                        help=f"Model's config for running evaluation. For example, see `data/configs`.")
    parser.add_argument('--dataset_path', '-d', type=str, required=True,
                        help="Dataset CSV path.")
    parser.add_argument('--output_file_name', '-o', type=str, default='output.json',
                        help="Output file name.")
    parser.add_argument("--debug", action="store_true", dest="debug",
                        help=f"Debug with {DEBUG_COUNT} samples.")
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    config_path = args.model_config_path.strip()
    model_config = {}
    if not os.path.exists(config_path):
        raise ValueError(f"Config path `{config_path}` does not exist!")
    else:
        with open(config_path, 'r') as f:
            model_config = json.load(f)

    # Check if need to initialize vLLM
    use_vllm = model_config.get('use_vllm', False)
    model_id = model_config.get('model_id')

    # Initialize vLLM model globally (once)
    model_id = model_config.get('model_id')
    tensor_parallel_size = model_config.get('tensor_parallel_size', torch.cuda.device_count())
    TOKENIZER = AutoTokenizer.from_pretrained(model_id)
    SAMPLING_PARAMS = SamplingParams(**model_config.get("generation_args", {}))
    
    MODEL = LLM(model_id, tensor_parallel_size=tensor_parallel_size, **model_config.get("model_args", {}))

    df = pd.read_csv(args.dataset_path)

    # Apply the formatting
    df['prompt'] = df.apply(lambda row: INSTRUCTION_FORMAT.format(title=row['title'], abstract=row['abstract']), axis=1)

    # Convert to list of messages
    messages = [[{"role": "user", "content": prompt}] for prompt in df['prompt']]
    dataset_ids = [id_ for id_ in df['id']]
    
    if args.debug:
        dataset_ids = dataset_ids[:DEBUG_COUNT]
        messages = messages[:DEBUG_COUNT]

    # Perform inference
    vllm_completion(messages, dataset_ids, args.output_file_name)
