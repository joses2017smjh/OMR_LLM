import json
from torch.utils.data import Dataset
from Linear import linearize  # reuse shared converter


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

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        src_text = ex.get('question_text') or ex.get('Problem')
        if src_text is None:
            raise KeyError(f"No question field found in example keys: {list(ex.keys())}")

        # Linearize the graph or use precomputed linear_formula
        if 'graph' in ex:
            lin_str = linearize(ex['graph'])
        else:
            lin_str = ex.get('linear_formula', '')
        if not lin_str:
            raise KeyError(f"No graph or linear_formula found in example keys: {list(ex.keys())}")

        src_enc = self.tokenizer_src(
            src_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_src_len,
            return_tensors='pt'
        )
        tgt_enc = self.tokenizer_tgt(
            lin_str,
            truncation=True,
            padding='max_length',
            max_length=self.max_tgt_len,
            return_tensors='pt'
        )

        return {
            'input_ids': src_enc.input_ids.squeeze(0),
            'attention_mask': src_enc.attention_mask.squeeze(0),
            'relation_labels': tgt_enc.input_ids.squeeze(0),
            'entity_labels': tgt_enc.input_ids.squeeze(0),
        }

    