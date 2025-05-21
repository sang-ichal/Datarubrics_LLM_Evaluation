import os
import logging
import json
import argparse
import glob

from .constants import *
from .utils import write_results
from .dataset_generator import create_dataset
from .models import generate_responses

def main():
    parser = argparse.ArgumentParser(description='Run inference on PDFs')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Provide the model config you want to use.')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Provide the input folder of jsonl(s).')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Provide the name of the output file.')
    parser.add_argument('--flush_size', type=int, default=-1,
                        help='When to flush the results out.')
    parser.add_argument("--debug", action="store_true", dest="debug",
                        help=f"Debug with {DEBUG_COUNT} samples")
    parser.set_defaults(debug=False)
    args = parser.parse_args()

    # Check config path
    config_path = args.model_config.strip()
    config_abs_path = os.path.join(ROOT_DIR, config_path)
    config = {}
    if not os.path.exists(config_abs_path):
        raise ValueError(f"Config path `{config_abs_path}` does not exist!")
    elif not config_abs_path.endswith('.json'):
        raise NotImplementedError("Config path is not in JSON Format, other format is not implemented yet!")
    else:
        with open(config_abs_path, 'r') as f:
            config = json.load(f)
        
        model_name = config.get('model_name', None)
        if model_name is None:
            logging.error(f"Config {config_abs_path} does not have `model_name` provided.")
            raise ValueError()
        elif model_name not in MODEL_LIST:
            logging.warning(f"Model {model_name} is not recognized! Defaulting to Transformer!")
            
    input_folder = args.input_folder
    if not os.path.exists(input_folder) or len(glob.glob(os.path.join(input_folder, "*.jsonl"))) < 1:
        raise ValueError(f"Input folder path `{input_folder}` does not contain any jsonl files!")

    # Create dataset and prompts
    output_path = os.path.join(ROOT_DIR, args.output_file)
    os.makedirs(os.path.abspath(os.path.dirname(output_path)), exist_ok=True)
    final_dataset = create_dataset(input_folder=input_folder, output_path=output_path,
                                   debug=args.debug)
    generate_responses(model_name=model_name, config=config,
                       final_dataset=final_dataset,
                       output_path=output_path, flush_size=args.flush_size,)

if __name__ == "__main__":
    main()