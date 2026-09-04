import os


# 1. Enforce air-gapped offline modes globally before importing frameworks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


import uvloop
from vllm.utils.argparse_utils import FlexibleArgumentParser
from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
from vllm.entrypoints.serve.utils.api_utils import cli_env_setup


if __name__ == "__main__":
    cli_env_setup()

    # 2. Hardcode configuration args to prevent unauthorized command line overrides
    parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )
    parser = make_arg_parser(parser)
    
    # Define exact private runtime specifications
    custom_args = [
        "--model", "~/Documents/models/Llama-3.2-3B-Instruct",
        "--host", "127.0.0.1",                       # Bind to localhost or specific internal IP
        "--port", "8000",
        "--api-key", "your-internal-secure-gateway-token-xyz", # Secure token authentication
        "--no-enable-log-requests",                   # Privacy setting: Never write prompts to logs
        "--enforce-eager",                           # Avoid CUDA graph memory overhead if needed
        "--gpu-memory-utilization", "0.85",          # Cap GPU utilization safely
        "--max-model-len", "2048",                   # Conservative context window for 6GB VRAM
        "--cpu-offload-gb", "3"                      # Offload ~3GB weights to system RAM to fit on 6GB GPU
    ]
    
    args = parser.parse_args(custom_args)
    validate_parsed_serve_args(args)
    
    # 3. Initialize and run the high-performance async inference engine
    uvloop.run(run_server(args))
