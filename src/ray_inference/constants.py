import os

# __file__ will be /path/to/repo/src/inference/constants.py
# os.path.dirname(__file__) is /path/to/repo/src/inference
# os.path.dirname(os.path.dirname(__file__)) is /path/to/repo/src
# os.path.dirname(os.path.dirname(os.path.dirname(__file__))) is /path/to/repo
CUR_DIR_CONSTANTS = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CUR_DIR_CONSTANTS, "..", "..")) # This should resolve to ./repo

# Validate ROOT_DIR for sanity, expecting "src" to be a child of ROOT_DIR
# If this assertion fails, the ROOT_DIR calculation needs adjustment based on your exact structure.
# assert os.path.basename(os.path.dirname(CUR_DIR_CONSTANTS)) == "src", \
#        f"constants.py might not be in the expected src/inference subdirectory. ROOT_DIR: {ROOT_DIR}"


DATA_DIR = os.path.join(ROOT_DIR, "data")
SCHEMA_DIR = os.path.join(DATA_DIR, "schema") # Assuming "schema" folder within "data"

CUR_DIR = CUR_DIR_CONSTANTS # For any code in this file that might use it.

OPENAI_RETRIES = 3
DEBUG_COUNT = 5
RANDOM_SEED = 42

# That is currently supported
MODEL_LIST = [
    "gpt-4.1-mini",
    "gemini-2.0-flash",
    "Qwen/Qwen3-4B", # Original Hugging Face ID, for local vLLM/Outlines via default_completion
    "Qwen/Qwen3-32B", # Original Hugging Face ID
    "microsoft/Phi-4-reasoning-plus",
    "qwen3-32b-12321", 
]