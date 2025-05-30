import json
import re
import torch
from torch.utils.data import Dataset
from Linear import linearize  # reuse shared converter

# 1a) load your MathQA ops & consts from disk
with open('./data/MathQA/operation_list.txt') as f:
    operator_list = [l.strip() for l in f if l.strip()]
with open('./data/MathQA/constant_list.txt') as f:
    constant_list = [l.strip() for l in f if l.strip()]

# 1b) build small‐vocab mapping
op2small = { op:i for i, op in enumerate(operator_list) }
pad_full_id = None  # we’ll fill this in __init__

class MathGraphDataset(Dataset):
    def __init__(self, path, tokenizer_src, tokenizer_tgt, max_src_len=128, max_tgt_len=64):
        """
        Dataset for MathQA: loads JSON entries containing 'question_text' and 'graph'.
        - tokenizer_src: HuggingFace tokenizer for source text
        - tokenizer_tgt: HuggingFace tokenizer for linearized target sequences
        """
        with open(path, 'r') as f:
            self.examples = json.load(f)
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        
        global pad_full_id
        pad_full_id = tokenizer_tgt.pad_token_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        src_text = ex.get('question_text') or ex['Problem']
        lin_str  = ex.get('linear_formula', '')

        # 1) source encoding (unchanged)
        src = self.tokenizer_src(
            src_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_src_len,
            return_tensors='pt'
        )
        input_ids      = src.input_ids.squeeze(0)
        attention_mask = src.attention_mask.squeeze(0)

        # 2) prepare empty label arrays
        pad_small_id = -100
        pad_full_id  = self.tokenizer_tgt.pad_token_id
        max_T = self.max_tgt_len

        rel_labels = torch.full((max_T,), pad_small_id, dtype=torch.long)
        ent_labels = torch.full((max_T,), pad_full_id,  dtype=torch.long)

        # 3) parse triplets and fill labels
        triplets = [t for t in lin_str.strip('|').split('|') if t]
        for i, trip in enumerate(triplets):
            base = 3*i
            if base+2 >= max_T:
                break
            m = re.match(r"(\w+)\(([^,]+),([^)]+)\)", trip)
            if not m:
                continue
            rel_tok, a1, a2 = m.group(1), m.group(2), m.group(3)

            # relation → small ID
            if rel_tok in op2small:
                rel_labels[base] = op2small[rel_tok]
            # entities → full‐vocab ID
            # full‐vocab IDs for a1,a2, but if unknown return pad_full_id
            eid1 = self.tokenizer_tgt.convert_tokens_to_ids(a1)
            eid2 = self.tokenizer_tgt.convert_tokens_to_ids(a2)
            # if the constant/placeholder wasn’t in your special_tokens => become PAD
            if eid1 == self.tokenizer_tgt.unk_token_id:
                eid1 = pad_full_id
            if eid2 == self.tokenizer_tgt.unk_token_id:
                eid2 = pad_full_id
            ent_labels[base+1] = eid1
            ent_labels[base+2] = eid2


        tgt = self.tokenizer_tgt(
            lin_str,
            truncation=True,
            padding="max_length",
            max_length=self.max_tgt_len,
            return_tensors="pt"
        )
        full_labels = tgt.input_ids.squeeze(0)
       
        return {
            "input_ids":       input_ids,
            "attention_mask":  attention_mask,
            "full_labels":     full_labels,
            "relation_labels": rel_labels,
            "entity_labels":   ent_labels,
    }

    
