import os
import logging
import json
import time
import tempfile
import concurrent.futures
from functools import partial
from collections import defaultdict

import torch.cuda
from tqdm import tqdm
from openai import OpenAI
import openai
from outlines import models, generate
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from .constants import *
from .utils import *

# Define for local models so only one time initialization
MODEL = None
TOKENIZER = None

def _request_openai_completion(openai_client, model_id, config, input_item):
    for attempt in range(OPENAI_RETRIES):
        try:
            response = openai_client.chat.completions.create(
                model=model_id,
                messages=input_item['msg'],
                response_format={
                    "type": "json_schema",
                    "json_schema": input_item['schema']
                },
                **config['generation_args']
            )
            
            result = {
                "id": input_item["id"],
                "token_usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "response": parse_json(response.choices[0].message.content)
            }
            return result
        except openai.OpenAIError as e:
            if "rate" in str(e).lower():
                logging.warning("Hit rate limit; retrying...")
                time.sleep(61)
            else:
                logging.exception("Error calling OpenAI API:")
                raise e
    logging.exception(f"Could not resolve error after {OPENAI_RETRIES} attempts for input id: {input_item['data_id']}")
    return None

def openai_completion(model_id, config, batched_input):
    """OpenAI completion which uses OpenAI client

    Args:
        model_id (str): Model's name
        config (dict): Model's configuration
        batched_input (list): List of input

    Returns:
        list[dict]: List of dictionary results (inputs with responses)
    """
    results = []
    
    if config.get('use_batch', False):
        # Prepare input file
        openai_client = OpenAI()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".jsonl", mode="w+", dir=CUR_DIR) as f:
            for input_item in batched_input:
                request_msg = {
                    "custom_id": input_item['id'],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_id,
                        "messages": input_item['msg'],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": input_item['schema']
                        },
                        **config['generation_args']
                    }
                }
                f.write(json.dumps(request_msg) + '\n')
                f.flush()

            batch_input_file = openai_client.files.create(
                file=open(f.name, "rb"),
                purpose="batch"
            )

            batch_input_file_id = batch_input_file.id
            async_batch = openai_client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": "Evaluation"
                }
            )
            batch_id = async_batch.id
            
            # Yes this is not async at the end of the day, we just want the batch
            batch_status = "in_progress"
            while True:
                batch = openai_client.batches.retrieve(batch_id)
                batch_status = batch.status
                if batch_status == "completed":
                    logging.info(f"Batch completed! Batch information: {batch}")
                    success_response = openai_client.files.content(batch.output_file_id).text
                    for line in success_response.split('\n')[:-1]:
                        parsed_data = json.loads(line)
                        response_id = parsed_data.get("custom_id")    
                        response = parsed_data["response"]["body"]["choices"][0]["message"]["content"]
                        
                        usage = parsed_data["response"]["body"].get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens")
                        completion_tokens = usage.get("completion_tokens")
                        total_tokens = usage.get("total_tokens")
                        
                        results.append({'id': response_id,
                                        "token_usage": {
                                            "prompt_tokens": prompt_tokens,
                                            "completion_tokens": completion_tokens,
                                            "total_tokens": total_tokens
                                        },
                                        "response": response})
                    break
                elif batch_status == "failed":
                    logging.warning(f"Batch failed with error: {batch}")
                    error_response = openai_client.files.content(batch.error_file_id).text
                    logging.warning(f"Decoded error response: {error_response}")
                    break
                elif batch_status == "cancelling":
                    logging.warning(f"Batch was cancelled: {batch}")
                    break
                else:
                    logging.info(f"Batch still in progress... Status: {batch.request_counts.completed}/{batch.request_counts.failed}/{batch.request_counts.total} (completed/failed/total) requests")
                    time.sleep(60)  # Avoid hitting rate limits, check every 1 mins
    else:
        if model_id.startswith("deepseek"):
            openai_client = OpenAI(base_url="https://api.deepseek.com")
        else:   
            openai_client = OpenAI()

        # Using ThreadPoolExecutor to process batched_input concurrently
        num_workers = config.get("num_workers", 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_input = {
                executor.submit(partial(_request_openai_completion, openai_client, model_id, config, input_item)): 
                input_item for input_item in batched_input
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_input), total=len(future_to_input), desc="OpenAI requests"):
                res = future.result()
                if res is not None:
                    results.append(res)
                        
    return results

def gemini_completion(model_id, config, batched_input):
    from google import genai
    from google.genai import types

    results = []
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    
    for input_item in batched_input:
        prompt = input_item['msg'][0]['content'] # Manually get the prompt
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt],
            config=types.GenerateContentConfig(
                **config.get("generation_args", {})
            )
        )
        results.append({"id": input_item['id'],
                        "response": response.text})
    return results


def default_completion(model_id: str, config: dict, batched_input: list) -> list[dict]:
    """Default completion which is either using VLLM or Transformers

    Args:
        model_id (str): Model's name
        config (dict): Model's configuration
        batched_input (list): List of input

    Returns:
        list[dict]: List of dictionary results (inputs with responses)
    """
    # Initialize model and tokenizer
    global MODEL, TOKENIZER
    if TOKENIZER is None:
        TOKENIZER = AutoTokenizer.from_pretrained(model_id, token=os.getenv('HF_TOKEN'))
    
    if MODEL is None:
        parallel_size = torch.cuda.device_count() if "tensor_parallel_size" not in config else config.get("tensor_parallel_size")
        MODEL = models.vllm(model_name=model_id, tensor_parallel_size=parallel_size, **config.get("model_args", {}))
    sampling_params = SamplingParams(**config.get("generation_args", {}))

    # Group by schema class
    schema_groups = defaultdict(list)
    for idx, item in enumerate(batched_input):
        schema_class = item["schema"]["name"]  # group by schema class
        schema_groups[schema_class].append((idx, item))  # keep index for reordering
        
     # Response placeholder
    response_list = [None] * len(batched_input)

    # Process each schema group
    for schema_class, group in schema_groups.items():
        indices, items = zip(*group)
        prompts = [
            TOKENIZER.apply_chat_template([item["msg"]],
                                          add_generation_prompt=True,
                                          tokenize=False)[0]
            for item in items
        ]
        generator = generate.json(MODEL, json.dumps(items[0]["schema"].get('schema')))  # use any schema instance from group
        outputs = generator(prompts, sampling_params=sampling_params)
        if not isinstance(outputs, list):
            outputs = [outputs]

        # Store outputs in correct global response list
        for idx, output in zip(indices, outputs):
            response_list[idx] = output

    # Final formatting
    results = [
        {
            "id": input_item["id"],
            "response": response
        }
        for input_item, response in zip(batched_input, response_list)
    ]

    return results

def generate_responses(model_id, config, final_dataset, output_path, flush_size):
    if flush_size <= 0:
        if 'gpt' in model_id or 'deepseek' in model_id:
            results = openai_completion(model_id, config, final_dataset)
        elif 'gemini' in model_id:
            results = gemini_completion(model_id, config, final_dataset)
        else:
            results = default_completion(model_id, config, final_dataset)
        write_results(results=results, output_path=output_path)
    else:
        for start_idx in tqdm(range(0, len(final_dataset), flush_size)):
            batched_question_data = final_dataset[start_idx: start_idx + flush_size]
            if 'gpt' in model_id or 'deepseek' in model_id:
                batched_results = openai_completion(model_id, config, batched_question_data)
            elif 'gemini' in model_id:
                batched_results = gemini_completion(model_id, config, batched_question_data)
            else:
                batched_results = default_completion(model_id, config, batched_question_data)
            write_results(results=batched_results, output_path=output_path)
