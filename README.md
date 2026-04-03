# K2-Lite evaluation based on Eval360

Eval360 is a long-context language model evaluation workspace built around the LM Evaluation Harness. It provides opinionated scripts and automation for benchmarking large checkpoints on reasoning, math, and code suites while coordinating large-cluster workflows (SLURM, Ray, and vLLM). The repository glues together local checkpoints, Hugging Face models, and multi-node serving endpoints to streamline end-to-end evaluation runs.


## Repository Layout
```
Eval360/
├── README.md
├── scripts/
│   ├── display/               # Result summarizers (python + notebooks)
│   ├── download/              # Model download utilities
│   ├── eval/                  # Base & instruct evaluation launchers
│   └── serving/               # vLLM + Ray Serve job scripts and clients
```

> The workspace includes additional submodules (`lm-evaluation-harness/`, `LOOM-Scope/`) that supply core evaluation logic and long-context benchmark suites.

## Prerequisites
- Access to a SLURM-based GPU cluster (scripts expect `sbatch`, multi-GPU nodes, and optional multi-node allocations).
- Python 3.10+ environment with CUDA-capable dependencies; a Miniconda/Conda install is assumed in job scripts.
- `lm_eval` (LM Evaluation Harness), `vllm`, `ray`, `fire`, and Hugging Face libraries installed.
- Hugging Face access token for gated models (`HF_TOKEN` in download scripts).
- Optional: OpenAI-compatible client libraries if calling serving endpoints through the provided API client.

## Environment Setup
1. **Clone with submodules**
   ```bash
   git clone --recursive <repo-url> K2-Lite
   cd K2-Lite
   ```
2. **Create environment** (example)
   ```bash
   conda create -n eval360 python=3.10
   conda activate eval360
   pip install -r lm-evaluation-harness/requirements.txt
   pip install vllm ray[serve] fire
   ```
3. **Environment variables**  
   Set these in your shell or SLURM scripts as needed:
   - `HF_ALLOW_CODE_EVAL=1` (enable code eval tasks)
   - `VLLM_WORKER_MULTIPROC_METHOD=spawn`
   - `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` (for long-context inference)
   - `HF_TOKEN=<token>` for gated Hugging Face downloads
   - Modify `PATH` to point at your conda install (examples already in scripts)

# K2-Lite-1.7B

## Downloading the Model

Download the model from HuggingFace:

```
https://huggingface.co/Amshaker/K2-Lite-1.7B
```

Place the model under `checkpoints/K2-Lite-1.7B`.

## Running Inference
 
The example below uses a math problem as the prompt and parses the model's thinking trace separately from its final answer.
 
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
 
model_name = "Amshaker/K2-Lite-1.7B"
 
# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
 
# Prepare the model input
prompt = "Solve: What is the sum of all integers from 1 to 100?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
 
# Conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=32768
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
 
# Parse thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0
 
thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
 
print("thinking content:", thinking_content)
print("content:", content)
```

## Running Evaluations

All evaluation scripts shell out to `lm_eval` with preset tasks.

```bash
MODEL_NAME=checkpoints/K2-Lite-1.7B

METRICS=(
    "arc_challenge_chat:90000:85000:1"
    "minerva_math500:90000:85000:1"
    "gpqa_diamond_cot_zeroshot:90000:85000:1"
    "gsm8k_reasoning_instruct:90000:85000:1"
    "aime24:90000:85000:8"
    "aime25:90000:85000:8"
    "minerva_math_reasoning_instruct:60000:55000:1"
)

for metric_config in "${METRICS[@]}"; do
    IFS=':' read -r METRIC_NAME MAX_LENGTH MAX_GEN_TOKENS N <<< "$metric_config"
    lm_eval --model vllm \
            --model_args pretrained=${MODEL_NAME},tensor_parallel_size=1,dtype=bfloat16,gpu_memory_utilization=0.9,max_length=${MAX_LENGTH} \
            --tasks ${METRIC_NAME} \
            --output_path logs/ \
            --batch_size auto \
            --apply_chat_template \
            --log_samples \
            --gen_kwargs do_sample=true,temperature=1.4,top_k=20,top_p=1.0,max_gen_toks=${MAX_GEN_TOKENS},n=${N}
done
```

### Benchmark Configuration

Each entry in `METRICS` follows the format `TASK:MAX_LENGTH:MAX_GEN_TOKENS:N`, where:

- **`TASK`** — `lm_eval` task name
- **`MAX_LENGTH`** — total context length (prompt + generation)
- **`MAX_GEN_TOKENS`** — maximum tokens to generate
- **`N`** — number of samples per problem (use `>1` for pass@k tasks like AIME)

### Evaluated Tasks

| Task | Max Length | Max Gen Tokens | Samples (N) |
|---|---|---|---|
| `arc_challenge_chat` | 90 000 | 85 000 | 1 |
| `minerva_math500` | 90 000 | 85 000 | 1 |
| `gpqa_diamond_cot_zeroshot` | 90 000 | 85 000 | 1 |
| `gsm8k_reasoning_instruct` | 90 000 | 85 000 | 1 |
| `aime24` | 90 000 | 85 000 | 8 |
| `aime25` | 90 000 | 85 000 | 8 |
| `minerva_math_reasoning_instruct` | 60 000 | 55 000 | 1 |

Results are saved to `logs/`.

## Acknowledgements
K2-Lite evaluation is based on Eval360, which builds on the open-source LM Evaluation Harness and leverages vLLM, Ray Serve, and various Hugging Face model releases. Review the respective licenses and documentation for details on redistribution and usage.
