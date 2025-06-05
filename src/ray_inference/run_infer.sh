#!/bin/bash

python3 -m src.ray_inference.inference --model_config data/configs/qwen3_32b_served_config.json --input_folder "/home/user/ray_projects/david_a/code/olmocr/datarubrics-dev-main/data/output/filtered_year_nlp_conference" --output_file nlp_out.json --flush_size 8192