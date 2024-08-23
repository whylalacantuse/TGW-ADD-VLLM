import traceback
from pathlib import Path

from vllm import LLM, SamplingParams
import torch

import modules.shared as shared
from modules.logging_colors import logger
from modules.text_generation import (
    get_max_prompt_length,
    get_reply_from_output_ids
)

try:
    import flash_attn
except ModuleNotFoundError:
    logger.warning(
        'You are running ExLlamaV2 without flash-attention. This will cause the VRAM usage '
        'to be a lot higher than it could be.\n'
        'Try installing flash-attention following the instructions here: '
        'https://github.com/Dao-AILab/flash-attention#installation-and-features'
    )
    pass
except Exception:
    logger.warning('Failed to load flash-attention due to the following error:\n')
    traceback.print_exc()

def load(model_name):
	path_to_model = Path(f'{shared.args.model_dir}/{model_name}')
	
	model = LLM(
		model= path_to_model,
		dtype="auto",
		trust_remote_code=True,
		tensor_parallel_size=1,
		gpu_memory_utilization=1,
	)
	tokenizer = model.get_tokenizer()

	stop_token_ids = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>")]
	generation_params = SamplingParams(
		temperature=0.5,
		top_p=0.8,
		top_k=50,
		repetition_penalty=1,
		stop_token_ids=stop_token_ids
	)

	input_ids = tokenizer.build_chat_input(query, history=[], role='user').input_ids[0].tolist()
	outputs = model.generate(
		sampling_params=generation_params,
		prompt_token_ids=[input_ids],
	)
	output = outputs[0]
	return output
		