import os
import json
import torch
from datasets import load_dataset

dirs = ['tests']

for d in dirs:
    os.makedirs(d, exist_ok=True)

# Attempt to load local MathQA JSON; if missing, pull HF math_qa dataset
try:
    ds_full = load_dataset(
        'json',
        data_files='data/mathqa_train.json',
        split='train'
    )
    print("Loaded local MathQA JSON (data/mathqa_train.json)")
except Exception:
    print("Local file not found, falling back to HF 'math_qa' dataset")
    ds_full = load_dataset(
        'math_qa',
        'all',
        split='train',
        trust_remote_code=True  # allow custom dataset code
    )

# Sample N examples
N = 10
ds_sample = ds_full.shuffle(seed=42).select(range(N))
sample_list = [dict(ex) for ex in ds_sample]
# Write sample to tests/temp_sample.json
sample_path = os.path.join('tests', 'temp_sample.json')
with open(sample_path, 'w') as f:
    json.dump(sample_list, f)
print(f"Wrote {N} samples to {sample_path}")

# Import your loader
from dataloader import MathGraphDataset
from Linear import linearize
try:
    from transformers import AutoTokenizer, LlamaTokenizer
except ImportError:
    raise RuntimeError("Please run `pip install transformers tokenizers` before testing.")

# Example: using your local Ollama LLaMA-2 model
src_tok = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.2-1B",
    trust_remote_code=True,
    token=True
)
tgt_tok = src_tok
# add operator/entity tokens used by linearizer

tgt_tok.add_tokens(['power', 'divide', 'sqrt', 'const0', 'const1', 'out0', 'out1'])

src_tok.pad_token    = src_tok.eos_token
src_tok.pad_token_id = src_tok.eos_token_id

tgt_tok.pad_token    = tgt_tok.eos_token
tgt_tok.pad_token_id = tgt_tok.eos_token_id
# Load dataset
ds = MathGraphDataset(
    path=sample_path,
    tokenizer_src=src_tok,
    tokenizer_tgt=tgt_tok,
    max_src_len=64,
    max_tgt_len=64
)

# Smoke-test first N items
for i in range(N):
    item = ds[i]
    assert isinstance(item['input_ids'], torch.Tensor)
    assert item['input_ids'].ndim == 1
    assert item['relation_labels'].ndim == 1
print(f"MathGraphDataset successfully processed {N} real examples!")
