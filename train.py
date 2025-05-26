import os
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
import wandb
from torcheval.metrics.functional import bleu_score
from datasets import load_dataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm



from dataloader import MathGraphDataset
from models.base import BaseModel
from models.rnn_model import EncoderDecoderRNN

# Configuration
config = {
    'project': 'MathQA-Graph-Translator-teacher_forcing_RNN',
    'bs': 32,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'embed_dim': 256,
    'hidden_dim': 512,
    'num_layers': 2,
    'max_epochs': 20,
    'max_src_len': 64,
    'max_tgt_len': 64

}


def dry_run(model_cls, config, device=None):
    """
    Quick sanity check: runs a batch of synthetic data through the model.
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    vocab_size = len(tokenizer)
    model = model_cls(
        src_vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        relation_vocab_size=vocab_size,
        entity_vocab_size=vocab_size
    ).to(device)

    batch_size = config['bs']
    src_len = config['max_src_len']
    tgt_len = config['max_tgt_len']

    # random source inputs
    input_ids = torch.randint(0, vocab_size, (batch_size, src_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    # dummy decoder input for teacher forcing: [SOS] + random
    decoder_input_ids = torch.full((batch_size, tgt_len), SOS_ID, dtype=torch.long, device=device)
    decoder_input_ids[:, 1:] = torch.randint(0, vocab_size, (batch_size, tgt_len-1), device=device)

    with torch.no_grad():
        rel_logits, ent_logits = model(
            input_ids,
            attention_mask,
            decoder_input_ids=decoder_input_ids
        )

    assert rel_logits.shape == (batch_size, tgt_len, vocab_size)
    assert ent_logits.shape == (batch_size, tgt_len, vocab_size)
    print(f"Dry run: rel={rel_logits.shape}, ent={ent_logits.shape}")

def evaluate(model, loader, tokenizer, device):
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
    with torch.no_grad():
        for batch in loader:
            ids = batch['input_ids'].to(device)
            att = batch['attention_mask'].to(device)
            labels = batch['relation_labels'].to(device)
            # use teacher-forcing path by passing labels as decoder_input_ids
            decoder_input = torch.full_like(labels, SOS_ID)
            decoder_input[:,1:] = labels[:,:-1]
            logits, _ = model(ids, att, decoder_input)
            b, t, v = logits.size()
            loss = criterion(logits.view(-1, v), labels.view(-1))
            preds = logits.argmax(dim=-1)
            mask = labels != tokenizer.pad_token_id
            total_loss += loss.item()
            total_correct += preds.masked_select(mask).eq(labels.masked_select(mask)).sum().item()
            total_tokens += mask.sum().item()
    return total_loss/total_tokens, total_correct/total_tokens


def compute_bleu(model, loader, tokenizer, device):
    model.eval()
    candidates, references = [], []

    with torch.no_grad():
        for batch in loader:
            ids     = batch['input_ids'].to(device)
            att     = batch['attention_mask'].to(device)
            labels  = batch['relation_labels'].to(device)

            # prepare decoder inputs (teacher forcing)
            decoder_input = torch.full_like(labels, SOS_ID)
            decoder_input[:,1:] = labels[:,:-1]

            rel_logits, _ = model(ids, att, decoder_input)
            preds = rel_logits.argmax(dim=-1)

            for i in range(preds.size(0)):
                cand_str = tokenizer.decode(preds[i].tolist(), skip_special_tokens=True)
                ref_str  = tokenizer.decode(labels[i].tolist(), skip_special_tokens=True)
                candidates.append(cand_str)
                references.append([ref_str])

    # try 4-gram, fallback to unigram if too short
    try:
        return bleu_score(candidates, references, n_gram=4)
    except ValueError:
        return bleu_score(candidates, references, n_gram=1)


if __name__ == '__main__':
    # 1) Initialize W&B
    wandb.login()
    wandb.init(project=config['project'], config=config)

    # 2) Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        trust_remote_code=True,
        token=True
    )
    tokenizer.add_special_tokens({'bos_token': '<SOS>'})
    tokenizer.add_tokens(['power','divide','sqrt','const0','const1','out0','out1'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    SOS_ID = tokenizer.bos_token_id

    # 3) Dry-run
    dry_run(EncoderDecoderRNN, config)

    # 4) Data loading (with fallback)
    data_path = 'data/MathQA/train.json'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Expected dataset at {data_path}, but it does not exist.")
    ds = MathGraphDataset(
        path=data_path,
        tokenizer_src=tokenizer,
        tokenizer_tgt=tokenizer,
        max_src_len=config['max_src_len'],
        max_tgt_len=config['max_tgt_len']
    )

    val_split = int(0.1 * len(ds))
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, list(range(val_split, len(ds)))),
        batch_size=config['bs'], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(ds, list(range(val_split))),
        batch_size=config['bs'], shuffle=False
    )

    # 5) Model, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EncoderDecoderRNN(
        src_vocab_size=len(tokenizer),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        relation_vocab_size=len(tokenizer),
        entity_vocab_size=len(tokenizer)
    ).to(device)
    
    
    # 5) Loss & optimizer just with regular adam
    #rel_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    #ent_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    #optimizer = optim.Adam(model.parameters(), lr=1e-3)

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    warmup_epochs = max(1, config['max_epochs'] // 10)
    linear_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config['max_epochs'] - warmup_epochs, eta_min=1e-5)
    scheduler = SequentialLR(optimizer, schedulers=[linear_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    # Training loop with teacher forcing and step-level logs
    num_steps = config['max_epochs'] * len(train_loader)
    iteration = 0
    pbar = tqdm(total=num_steps, desc='Training Iterations', unit='batch')
    best_val_loss = float('inf')

    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        # log LR at start of epoch
        wandb.log({'LR': scheduler.get_last_lr()[0]}, step=iteration)

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            rel_labels = batch['relation_labels'].to(device)

            # prepare decoder inputs (shifted right with SOS)
            decoder_input_ids = torch.full(
                (rel_labels.size(0), config['max_tgt_len']),
                SOS_ID, dtype=torch.long, device=device
            )
            decoder_input_ids[:,1:] = rel_labels[:,:-1]

            optimizer.zero_grad()
            rel_logits, _ = model(input_ids, attention_mask, decoder_input_ids)

            # compute token-level loss
            b, t, v = rel_logits.size()
            criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')
            loss = criterion(rel_logits.view(-1, v), rel_labels.view(-1))
            loss.backward()
            optimizer.step()

            # accuracy
            preds = rel_logits.argmax(dim=-1)
            mask = rel_labels != tokenizer.pad_token_id
            acc = (preds == rel_labels).masked_select(mask).sum().float() / mask.sum().float()

            # log train metrics per step
            wandb.log({
                'Loss/train': loss.item() / mask.sum().item(),
                'Acc/train': acc.item(),
            }, step=iteration)

            iteration += 1
            scheduler.step()
            pbar.update(1)

        # end of epoch: validation
        model.eval()
        val_loss, val_acc = evaluate(model, val_loader, tokenizer, device)
        val_bleu = compute_bleu(model, val_loader, tokenizer, device)
        wandb.log({
            'Loss/val': val_loss,
            'Acc/val': val_acc,
            'BLEU/val': val_bleu,
        }, step=iteration)

        # checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(wandb.run.dir, 'best_model.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model at epoch {epoch} (val_loss={val_loss:.4f})")

        pbar.close()

    wandb.finish()


    