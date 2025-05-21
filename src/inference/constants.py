import os

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(CUR_DIR)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SCHEMA_DIR = os.path.join(DATA_DIR, "schema")

OPENAI_RETRIES = 3
DEBUG_COUNT = 5
RANDOM_SEED = 42

# That is currently supported
MODEL_LIST = [
    "gpt-4.1-mini",
    "gemini-2.0-flash",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-32B",
    "microsoft/Phi-4-reasoning-plus",
]

