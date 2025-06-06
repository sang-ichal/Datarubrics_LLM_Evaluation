import os
import logging
import json
import argparse
import glob

# Assuming constants.py, utils.py, dataset_generator.py, models.py
# are in the same package or PYTHONPATH is set up correctly.
from .constants import ROOT_DIR, DEBUG_COUNT, MODEL_LIST # MODEL_LIST for validation
from .dataset_generator import create_dataset
from .ray_models import generate_responses

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Run inference on P PDFs (processed as JSONL inputs)')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to the model JSON configuration file (relative to ROOT_DIR or absolute).')
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Path to the input folder containing JSONL file(s) (relative to ROOT_DIR or absolute).')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Name of the output file (will be saved in ROOT_DIR, e.g., results/output.json).')
    parser.add_argument('--dataset_start_idx', type=int, default=0,
                        help='Dataset start index for slicing.')
    parser.add_argument('--dataset_end_idx', type=int, default=-1,
                        help='Dataset end index for slicing (-1 for up to the end).')
    parser.add_argument('--flush_size', type=int, default=100, # Default to flushing every 100 items
                        help='Number of items to process before flushing results to disk. <=0 means write all at the end.')
    parser.add_argument("--combine_prompts", action="store_true",
                        help="Whether to combine multiple rubrics into a single prompt.")
    parser.add_argument("--debug", action="store_true",
                        help=f"Run in debug mode with only {DEBUG_COUNT} samples.")
    args = parser.parse_args()

    # Resolve config path (can be absolute or relative to ROOT_DIR)
    config_path_arg = args.model_config.strip()
    if not os.path.isabs(config_path_arg):
        config_abs_path = os.path.join(ROOT_DIR, config_path_arg)
    else:
        config_abs_path = config_path_arg
    
    config = {}
    if not os.path.exists(config_abs_path):
        logger.error(f"Model config path `{config_abs_path}` does not exist!")
        raise FileNotFoundError(f"Model config path `{config_abs_path}` does not exist!")
    elif not config_abs_path.endswith('.json'):
        logger.error("Model config path must be a JSON file.")
        raise ValueError("Model config path is not in JSON Format.")
    else:
        logger.info(f"Loading model configuration from: {config_abs_path}")
        with open(config_abs_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        model_id_from_config = config.get('model_id')
        if not model_id_from_config:
            logger.error(f"Config {config_abs_path} must contain a 'model_id' field.")
            raise ValueError("'model_id' not found in config.")
        
        # Validate model_id against MODEL_LIST if it's not a generic API type that bypasses this list
        api_type = config.get("api_type")
        # For vLLM served models, model_id_from_config is the "served-model-name"
        # For HF local models, it's the repo ID.
        # This check is a sanity check that we have at least heard of this model.
        if model_id_from_config not in MODEL_LIST:
            logger.warning(
                f"Model ID '{model_id_from_config}' from config is not in the predefined MODEL_LIST in constants.py. "
                f"Ensure this is intended and the model server/setup is correct."
            )
            # Depending on strictness, you might raise an error here or allow it.
            # For now, just a warning.
            
    # Resolve input folder path
    input_folder_arg = args.input_folder.strip()
    if not os.path.isabs(input_folder_arg):
        input_folder_abs_path = os.path.join(ROOT_DIR, input_folder_arg)
    else:
        input_folder_abs_path = input_folder_arg

    if not os.path.exists(input_folder_abs_path) or not glob.glob(os.path.join(input_folder_abs_path, "*.jsonl")):
        logger.error(f"Input folder path `{input_folder_abs_path}` does not exist or contains no .jsonl files!")
        raise FileNotFoundError(f"Input folder `{input_folder_abs_path}` not valid or empty of .jsonl files.")

    # Output path is relative to ROOT_DIR
    # e.g., if args.output_file is "results/my_output.json", it's saved in ROOT_DIR/results/my_output.json
    output_file_rel_path = args.output_file 
    # The write_results function in utils.py will join this with ROOT_DIR.

    logger.info(f"Preparing dataset from: {input_folder_abs_path}")
    # dataset_generator.create_dataset expects output_path to be relative to ROOT_DIR for existing ID check
    final_dataset = create_dataset(input_folder=input_folder_abs_path, 
                                   output_path=output_file_rel_path, # Pass relative path for consistency
                                   start_idx=args.dataset_start_idx, 
                                   end_idx=args.dataset_end_idx,
                                   combine_prompts=args.combine_prompts, 
                                   debug=args.debug)
    
    if not final_dataset:
        logger.info("No new data to process after filtering existing IDs or due to dataset limits.")
        return

    logger.info(f"Starting response generation for {len(final_dataset)} items using model: {config.get('model_id')}")
    generate_responses(model_id_from_config=config['model_id'], # Pass the model_id from config
                       config=config,
                       final_dataset=final_dataset,
                       output_path=output_file_rel_path, # Pass relative path
                       flush_size=args.flush_size)
    
    logger.info(f"Inference run completed. Output saved to {os.path.join(ROOT_DIR, output_file_rel_path)}")

if __name__ == "__main__":
    # This allows running the script as `python -m your_package.inference ...`
    # or `python path/to/inference.py ...` if the package structure is handled by PYTHONPATH.
    main()