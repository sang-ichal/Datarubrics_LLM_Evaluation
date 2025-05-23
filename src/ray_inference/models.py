import os
import logging
import json
import time
import tempfile
import concurrent.futures
from functools import partial
from collections import defaultdict

import torch # Keep for torch.cuda.device_count()
import openai # Ensure openai library is version 1.0.0 or higher
from openai import OpenAI # Explicit import
from tqdm import tqdm

# For default_completion (local vLLM/HF with Outlines)
from outlines import models as outlines_models # Renamed to avoid conflict
from outlines import generate as outlines_generate
from vllm import LLM, SamplingParams # For default_completion if using vLLM directly via outlines
from transformers import AutoTokenizer # For default_completion

# Assuming constants.py and utils.py are in the same package directory
from .constants import CUR_DIR, OPENAI_RETRIES
from .utils import parse_json, write_results # write_results is used in generate_responses

# Global variables for default_completion (local Outlines/vLLM setup)
LOCAL_VLLM_MODEL = None
LOCAL_VLLM_TOKENIZER = None

httpx_logger = logging.getLogger("httpx")
httpx_logger.setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)


def _request_openai_completion(openai_client, model_id_for_server, config, input_item):
    """
    Helper function to make a single request to an OpenAI-compatible API.
    model_id_for_server: The model name as the server expects it (e.g., "Qwen3-32B-Chat-8100").
    """
    for attempt in range(OPENAI_RETRIES):
        try:
            response = openai_client.chat.completions.create(
                model=model_id_for_server,
                messages=input_item['msg'],
                response_format={"type": "json_schema", "json_schema": input_item['schema']},
                extra_body=config.get('extra_body', {}), 
                **config.get('generation_args', {})
            )
            
            token_usage = response.usage
            result = {
                "id": input_item["id"],
                "token_usage": {
                    "prompt_tokens": token_usage.prompt_tokens if token_usage else None,
                    "completion_tokens": token_usage.completion_tokens if token_usage else None,
                    "total_tokens": token_usage.total_tokens if token_usage else None,
                },
                "response": parse_json(response.choices[0].message.content)
            }
            return result
        except openai.APIError as e: # Catching more specific OpenAI errors
            error_message = str(e).lower()
            item_id = input_item.get('id', 'unknown_id')
            if "rate limit" in error_message or "quota" in error_message or "limit" in error_message:
                logger.warning(f"Rate limit/quota error for ID {item_id} (attempt {attempt + 1}/{OPENAI_RETRIES}): {e}. Retrying after delay...")
                time.sleep(60 * (attempt + 1)) # Exponential backoff for rate limits
            elif "connection" in error_message or "timeout" in error_message:
                logger.warning(f"Connection/timeout error for ID {item_id} (attempt {attempt + 1}/{OPENAI_RETRIES}): {e}. Retrying after delay...")
                time.sleep(30 * (attempt + 1))
            else:
                logger.exception(f"Unhandled OpenAI API error for ID {item_id} (attempt {attempt + 1}/{OPENAI_RETRIES}): {e}")
                # For unhandled errors, decide if retry is useful or if it should fail fast for this item
                if attempt == OPENAI_RETRIES - 1: # Last attempt
                    return {"id": item_id, "error": str(e), "response": None} # Return error for this item
                time.sleep(10) # Short delay before retrying other API errors
        except Exception as e: # Catch any other unexpected errors
            item_id = input_item.get('id', 'unknown_id')
            logger.exception(f"Unexpected error during API request for ID {item_id} (attempt {attempt + 1}/{OPENAI_RETRIES}): {e}")
            if attempt == OPENAI_RETRIES - 1:
                return {"id": item_id, "error": f"Unexpected error: {str(e)}", "response": None}
            time.sleep(10)

    item_id = input_item.get('id', 'unknown_id')
    logger.error(f"Could not resolve API request for ID {item_id} after {OPENAI_RETRIES} attempts.")
    return {"id": item_id, "error": f"Failed after {OPENAI_RETRIES} retries", "response": None}


def openai_compatible_completion(model_id_in_config, config, batched_input):
    """
    Completion for any OpenAI-compatible API, including OpenAI, DeepSeek, or a local vLLM server.
    model_id_in_config: The model identifier from the config file. For vLLM, this MUST be the "served-model-name".
    """
    results = []
    api_type = config.get("api_type", "openai") # Default to "openai" if not specified

    if api_type == "vllm_openai_compatible":
        base_url = config.get("base_url")
        if not base_url:
            logger.error("base_url not provided in config for vllm_openai_compatible model.")
            raise ValueError("base_url is required for vllm_openai_compatible")
        api_key = config.get("api_key", "EMPTY") # vLLM default
        openai_client = OpenAI(base_url=base_url, api_key=api_key)
        logger.info(f"Using vLLM OpenAI-compatible server at {base_url} for model '{model_id_in_config}'")
    elif model_id_in_config.startswith("deepseek"): # model_id_in_config here is the actual model name like "deepseek-coder"
        openai_client = OpenAI(base_url="https://api.deepseek.com", api_key=os.environ.get("DEEPSEEK_API_KEY"))
        logger.info(f"Using DeepSeek API for model '{model_id_in_config}'")
    else: # Default to official OpenAI API
        openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        logger.info(f"Using OpenAI API for model '{model_id_in_config}'")

    use_openai_batch_api = config.get('use_batch', False)
    if api_type == "vllm_openai_compatible" and use_openai_batch_api:
        logger.warning("OpenAI batch processing mode is not supported for 'vllm_openai_compatible'. Switching to concurrent requests.")
        use_openai_batch_api = False
    
    # The model_id_for_server is the name the API endpoint expects. For OpenAI/Deepseek, it's model_id_in_config.
    # For vLLM, model_id_in_config *is* the served-model-name.
    model_id_for_server = model_id_in_config

    if use_openai_batch_api and api_type != "vllm_openai_compatible":
        # OpenAI's own Batch API logic (mostly unchanged from your previous version)
        # Ensure input_item['id'] is used for custom_id.
        # ... (previous batch logic - keeping it concise here for brevity, use your full version)
        # Ensure this part is robust for actual OpenAI batching.
        logger.info("Using OpenAI Batch API processing.")
        # Placeholder for actual batch API call logic
        # For each item, if batch fails, should still try to add an error placeholder
        temp_jsonl_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl", mode="w+", dir=CUR_DIR) as f:
                temp_jsonl_path = f.name
                for item_idx, input_item in enumerate(batched_input):
                    custom_id = f"{input_item['id']}_{item_idx}" # Ensure unique custom_id
                    request_msg = {
                        "custom_id": custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": model_id_for_server,
                            "messages": input_item['msg'],
                            "response_format": {"type": "json_schema", "json_schema": input_item['schema']},
                            **config.get('generation_args', {})
                        }
                    }
                    f.write(json.dumps(request_msg) + '\n')
            
            with open(temp_jsonl_path, "rb") as file_to_upload:
                batch_file_obj = openai_client.files.create(file=file_to_upload, purpose="batch")
            
            batch_job = openai_client.batches.create(
                input_file_id=batch_file_obj.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={"description": f"Eval run for {model_id_for_server}"}
            )
            logger.info(f"OpenAI Batch job {batch_job.id} created. Waiting for completion (up to 24h)...")

            # Simplified polling - refer to OpenAI docs for robust polling/webhooks
            while batch_job.status not in ["completed", "failed", "cancelled"]:
                time.sleep(60)
                batch_job = openai_client.batches.retrieve(batch_job.id)
                logger.info(f"Batch {batch_job.id} status: {batch_job.status} ({batch_job.request_counts.completed}/{batch_job.request_counts.total} done)")

            if batch_job.status == "completed":
                logger.info(f"Batch {batch_job.id} completed.")
                if batch_job.output_file_id:
                    output_content = openai_client.files.content(batch_job.output_file_id).text
                    for line in output_content.splitlines():
                        if not line.strip(): continue
                        try:
                            data = json.loads(line)
                            original_custom_id = data.get("custom_id")
                            # Find corresponding input_item by custom_id logic if needed, or assume order.
                            # For simplicity, find original ID part.
                            original_id = original_custom_id.rsplit('_',1)[0] if '_' in original_custom_id else original_custom_id

                            response_body = data.get("response", {}).get("body", {})
                            content = response_body.get("choices", [{}])[0].get("message", {}).get("content")
                            usage = response_body.get("usage", {})
                            results.append({
                                'id': original_id,
                                "token_usage": {
                                    "prompt_tokens": usage.get("prompt_tokens"),
                                    "completion_tokens": usage.get("completion_tokens"),
                                    "total_tokens": usage.get("total_tokens")
                                },
                                "response": parse_json(content) if content else None
                            })
                        except Exception as e:
                            logger.error(f"Error processing batch output line: {line}. Error: {e}")
                if batch_job.error_file_id:
                     logger.error(f"Batch job {batch_job.id} had errors. Error file: {batch_job.error_file_id}")
                     # Optionally download and log errors
            else: # Failed or cancelled
                logger.error(f"Batch job {batch_job.id} ended with status: {batch_job.status}.")
                # Add error placeholders for all items in this batch
                for item in batched_input:
                    results.append({"id": item["id"], "error": f"OpenAI Batch job {batch_job.status}", "response": None})

        except Exception as e:
            logger.exception(f"Error during OpenAI Batch API process: {e}")
            for item in batched_input: # Add error placeholders
                 results.append({"id": item["id"], "error": f"OpenAI Batch API main error: {str(e)}", "response": None})
        finally:
            if temp_jsonl_path and os.path.exists(temp_jsonl_path):
                os.remove(temp_jsonl_path)
    else: # Concurrent requests (for vLLM, or if OpenAI batch is disabled)
        num_workers = config.get("num_workers", min(32, os.cpu_count() + 4)) # Default for I/O bound
        logger.info(f"Using ThreadPoolExecutor with {num_workers} workers for model '{model_id_for_server}'.")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_input = {
                executor.submit(
                    _request_openai_completion, 
                    openai_client, 
                    model_id_for_server, # This is the served model name for vLLM
                    config, 
                    input_item
                ): input_item 
                for input_item in batched_input
            }
            for future in tqdm(concurrent.futures.as_completed(future_to_input), total=len(future_to_input), desc=f"API requests for {model_id_for_server}"):
                try:
                    res = future.result()
                    if res is not None:
                        results.append(res)
                except Exception as exc:
                    failed_input_item = future_to_input[future]
                    logger.error(f"Request for ID {failed_input_item.get('id','unknown_id')} generated an exception in thread: {exc}")
                    results.append({"id": failed_input_item.get('id','unknown_id'), "error": str(exc), "response": None})
                        
    return results


def gemini_completion(model_id_in_config, config, batched_input):
    from google import genai # Import moved inside
    from google.generativeai import types as genai_types # Alias to avoid conflict if any

    results = []
    # API key should be set as an environment variable GOOGLE_API_KEY or configured in genai
    # genai.configure(api_key=os.environ["GOOGLE_API_KEY"]) # Typically done once at app start
    
    logger.info(f"Using Gemini API for model '{model_id_in_config}'")
    gemini_model = genai.GenerativeModel(model_name=model_id_in_config)
    
    gen_args = config.get("generation_args", {})
    # Gemini uses different names for some parameters, e.g., max_output_tokens
    if "max_tokens" in gen_args and "max_output_tokens" not in gen_args:
        gen_args["max_output_tokens"] = gen_args.pop("max_tokens")
    if "n" in gen_args and "candidate_count" not in gen_args: # 'n' for OpenAI, 'candidate_count' for Gemini
        gen_args["candidate_count"] = gen_args.pop("n")
    
    gemini_config = genai_types.GenerationConfig(**gen_args)

    for input_item in tqdm(batched_input, desc=f"Gemini requests for {model_id_in_config}"):
        try:
            prompt = input_item['msg'][0]['content'] # Assuming user message is the first one
            # For JSON with Gemini, the prompt must explicitly ask for JSON output.
            # Gemini doesn't have a 'response_format' like OpenAI's JSON mode.
            # The schema from input_item['schema'] is a hint that JSON is expected.
            
            response = gemini_model.generate_content(
                contents=[prompt],
                generation_config=gemini_config
            )
            
            # Gemini's response.text directly contains the output.
            # If there are multiple candidates, response.candidates[0].content.parts[0].text
            # For simplicity, assuming one candidate and text part.
            output_text = response.text if hasattr(response, 'text') else None
            if not output_text and response.candidates:
                 output_text = response.candidates[0].content.parts[0].text


            # Token count: response.usage_metadata (prompt_token_count, candidates_token_count, total_token_count)
            usage = response.usage_metadata if hasattr(response, 'usage_metadata') else None
            results.append({
                "id": input_item["id"],
                "token_usage": {
                    "prompt_tokens": usage.prompt_token_count if usage else None,
                    "completion_tokens": usage.candidates_token_count if usage else None, # Sum if multiple candidates
                    "total_tokens": usage.total_token_count if usage else None
                },
                "response": parse_json(output_text) if output_text else None
            })
        except Exception as e:
            logger.error(f"Error during Gemini completion for ID {input_item['id']}: {e}")
            results.append({"id": input_item["id"], "error": str(e), "response": None})
            
    return results


def default_hf_completion(model_id_in_config: str, config: dict, batched_input: list) -> list[dict]:
    """
    Default completion for HuggingFace models run locally using Outlines with vLLM backend.
    model_id_in_config: The Hugging Face model identifier (e.g., "Qwen/Qwen3-4B").
    """
    global LOCAL_VLLM_MODEL, LOCAL_VLLM_TOKENIZER
    
    if LOCAL_VLLM_TOKENIZER is None or LOCAL_VLLM_MODEL is None or \
       (hasattr(LOCAL_VLLM_MODEL, 'model_name') and LOCAL_VLLM_MODEL.model_name != model_id_in_config): # Re-init if model changed
        logger.info(f"Initializing local tokenizer for '{model_id_in_config}' via Outlines/HF.")
        LOCAL_VLLM_TOKENIZER = AutoTokenizer.from_pretrained(model_id_in_config, token=os.getenv('HF_TOKEN'))
        
        logger.info(f"Initializing local vLLM engine for '{model_id_in_config}' via Outlines.")
        
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        default_tp_size = num_gpus if num_gpus > 0 else 1
        # Allow config to override tensor_parallel_size
        tp_size_from_config = config.get("model_args", {}).get("tensor_parallel_size")
        if tp_size_from_config is not None:
            parallel_size = tp_size_from_config
        else: # Fallback to config top-level, then default
             parallel_size = config.get("tensor_parallel_size", default_tp_size)


        if num_gpus > 0 and parallel_size > num_gpus:
            logger.warning(f"Requested tensor_parallel_size ({parallel_size}) > available GPUs ({num_gpus}). Setting to {num_gpus}.")
            parallel_size = num_gpus
        elif num_gpus == 0 and parallel_size > 1:
            logger.warning(f"CUDA not available but tensor_parallel_size ({parallel_size}) > 1. Setting to 1 (CPU).")
            parallel_size = 1
        
        model_args_from_config = config.get("model_args", {})
        if "tensor_parallel_size" not in model_args_from_config : # Ensure it's set if not passed explicitly
             model_args_from_config["tensor_parallel_size"] = parallel_size
        if "max_model_len" not in model_args_from_config: # Default if not provided
             model_args_from_config["max_model_len"] = config.get("max_model_len", 8192)


        # Use outlines.models.vllm to load the model
        LOCAL_VLLM_MODEL = outlines_models.vllm(model_name=model_id_in_config, **model_args_from_config)
        LOCAL_VLLM_MODEL.model_name = model_id_in_config # Store for checking if model changed

    # vLLM SamplingParams from generation_args in config
    vllm_sampling_config = config.get("generation_args", {})
    # Parameters like 'response_format' are not for vLLM SamplingParams
    vllm_sampling_config.pop('response_format', None) 
    sampling_params = SamplingParams(**vllm_sampling_config)

    # Group inputs by schema for Outlines, as generator is schema-specific
    schema_groups = defaultdict(list)
    for idx, item in enumerate(batched_input):
        # Use a string representation of the schema for grouping if schemas can vary
        # For simplicity, using schema "name" if available, else hash of schema structure.
        schema_key = item["schema"].get("name", hash(json.dumps(item["schema"].get("schema", item["schema"]), sort_keys=True)))
        schema_groups[schema_key].append((idx, item))
        
    final_results_list = [None] * len(batched_input) # Placeholder for ordered results

    for schema_key, group_items in tqdm(schema_groups.items(), desc=f"Local HF/Outlines for {model_id_in_config}"):
        original_indices, items_in_group = zip(*group_items)
        
        prompts = []
        for item in items_in_group:
            if isinstance(item["msg"], list) and all(isinstance(m, dict) for m in item["msg"]):
                try:
                    prompts.append(LOCAL_VLLM_TOKENIZER.apply_chat_template(item["msg"], add_generation_prompt=True, tokenize=False))
                except Exception as e:
                    logger.error(f"Error applying chat template for item ID {item['id']}: {e}. Using raw content.")
                    prompts.append(item['msg'][0]['content'] if item['msg'] and item['msg'][0].get('content') else "")
            elif isinstance(item["msg"], str): # If msg is already a string
                 prompts.append(item["msg"])
            else:
                logger.warning(f"Unexpected message format for item ID {item['id']}. Skipping.")
                prompts.append("") # Add empty prompt to maintain list length

        # Get the JSON schema structure for Outlines
        # It expects the actual schema dictionary, not the whole schema object from the file
        json_schema_for_outlines = items_in_group[0]["schema"].get('schema', items_in_group[0]["schema"])
        
        try:
            # Create Outlines generator for the current schema
            generator = outlines_generate.json(LOCAL_VLLM_MODEL, json_schema_for_outlines)
            # Generate responses for all prompts in the current group
            generated_outputs_text = generator(prompts, sampling_params=sampling_params)
            
            if not isinstance(generated_outputs_text, list): # Ensure it's a list
                generated_outputs_text = [generated_outputs_text]

            for i, output_text in enumerate(generated_outputs_text):
                original_idx = original_indices[i]
                try:
                    parsed_response = parse_json(output_text)
                    # Note: Outlines + vLLM typically doesn't give token usage directly per request easily
                    final_results_list[original_idx] = {"id": items_in_group[i]["id"], "response": parsed_response}
                except Exception as e:
                    logger.error(f"Failed to parse JSON from Outlines output for ID {items_in_group[i]['id']}: {e}. Output: {output_text[:100]}")
                    final_results_list[original_idx] = {"id": items_in_group[i]["id"], "error": f"JSON parse error: {str(e)}", "response": None, "raw_output": output_text}
        except Exception as e:
            logger.exception(f"Error during local Outlines generation for model {model_id_in_config}, schema key {schema_key}: {e}")
            for i, original_idx in enumerate(original_indices): # Mark all items in this failed group
                final_results_list[original_idx] = {"id": items_in_group[i]["id"], "error": f"Outlines generation failed: {str(e)}", "response": None}
    
    return [res for res in final_results_list if res is not None] # Filter out any None placeholders if some items were skipped


def generate_responses(model_id_from_config: str, config: dict, final_dataset: list, output_path: str, flush_size: int):
    """
    Generates responses for the dataset using the specified model and configuration.
    model_id_from_config: The primary identifier for the model, from the JSON config.
                         For vLLM served models, this should be the 'served-model-name'.
                         For HF models for default_hf_completion, this is the HF repo ID.
    """
    api_type = config.get("api_type") # e.g., "vllm_openai_compatible", "gemini", or None for default/OpenAI
    
    # Determine effective flush_size for batching loop
    current_flush_size = flush_size if flush_size > 0 else len(final_dataset)
    if current_flush_size == 0 and len(final_dataset) > 0 : # Handle edge case where flush_size=0 but dataset exists
        current_flush_size = len(final_dataset)
    elif len(final_dataset) == 0:
        logger.info("Input dataset is empty. Nothing to process.")
        return


    for i in tqdm(range(0, len(final_dataset), current_flush_size), desc="Processing dataset in batches"):
        batched_data_segment = final_dataset[i : i + current_flush_size]
        if not batched_data_segment: # Should not happen if loop condition is correct
            continue

        logger.info(f"Processing segment from index {i} to {i + len(batched_data_segment) -1 } for model '{model_id_from_config}'")
        
        current_batch_results = []
        # Route to the correct completion function
        if api_type == "vllm_openai_compatible":
            logger.info(f"Using OpenAI-compatible completion for vLLM served model: {model_id_from_config}")
            current_batch_results = openai_compatible_completion(model_id_from_config, config, batched_data_segment)
        elif api_type == "gemini":
            logger.info(f"Using Gemini completion for model: {model_id_from_config}")
            current_batch_results = gemini_completion(model_id_from_config, config, batched_data_segment)
        elif model_id_from_config.startswith("gpt-") or model_id_from_config.startswith("deepseek") or api_type == "openai": # Default to OpenAI or Deepseek via compatible API
            logger.info(f"Using OpenAI-compatible completion for model: {model_id_from_config}")
            current_batch_results = openai_compatible_completion(model_id_from_config, config, batched_data_segment)
        else: # Fallback to local HuggingFace models with Outlines
            logger.info(f"Using default HuggingFace (Outlines/vLLM local) completion for model: {model_id_from_config}")
            # model_id_from_config here should be the Hugging Face repo ID like "Qwen/Qwen3-4B"
            current_batch_results = default_hf_completion(model_id_from_config, config, batched_data_segment)
        
        if current_batch_results:
            # write_results expects a list of dicts.
            # If flush_size > 0, append. Otherwise, it will be a single write at the end if logic outside loop handles it.
            # For simplicity, always append if flush_size active, or single write if not.
            is_append_mode = True if flush_size > 0 and i > 0 else False # Append if not the first flushed batch
            if flush_size <= 0: is_append_mode = False # Single write if no flushing.

            write_results(results=current_batch_results, output_path=output_path, append=is_append_mode)
            logger.info(f"Flushed {len(current_batch_results)} results to {output_path} (Append: {is_append_mode})")
        else:
            logger.info(f"No results generated for segment starting at index {i} for model '{model_id_from_config}'.")
    
    logger.info(f"All segments processed for model '{model_id_from_config}'. Final results in {output_path}")