import os
import json

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import wandb
from torcheval.metrics.functional import bleu_score
from datasets import load_dataset

from dataloader import MathGraphDataset
from models.rnn_model import EncoderDecoderRNN

# Configuration
config = {
    'project': 'MathQA-Graph-Translator',
    'bs': 32,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'embed_dim': 256,
    'hidden_dim': 512,
    'num_layers': 2,
    'max_epochs': 50,
    'max_src_len': 64,
    'max_tgt_len': 64,
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
    seq_len = config['max_src_len']
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        rel_logits, ent_logits = model(input_ids, attention_mask)

    assert rel_logits.shape == (batch_size, seq_len, vocab_size)
    assert ent_logits.shape == (batch_size, seq_len, vocab_size)
    print(f"Dry run successful: rel={rel_logits.shape}, ent={ent_logits.shape}")


def evaluate(model, loader, tokenizer, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='sum')

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            rel_labels = batch['relation_labels'].to(device)

            rel_logits, _ = model(input_ids, attention_mask)
            b, t, v = rel_logits.size()
            loss = criterion(rel_logits.view(-1, v), rel_labels.view(-1))

            preds = rel_logits.argmax(dim=-1)
            mask = rel_labels != tokenizer.pad_token_id
            correct = (preds == rel_labels) & mask

            total_loss += loss.item()
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy


def compute_bleu(model, loader, tokenizer, device):
    model.eval()
    candidates, references = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ref_labels = batch['relation_labels'].to(device)

            rel_logits, _ = model(input_ids, attention_mask)
            preds = rel_logits.argmax(dim=-1)

            for i in range(preds.size(0)):
                pred_ids = preds[i].tolist()
                ref_ids  = ref_labels[i].tolist()
                # strip pads
                pred_tokens = tokenizer.decode(pred_ids, skip_special_tokens=True).split()
                ref_tokens  = tokenizer.decode(ref_ids,  skip_special_tokens=True).split()
                candidates.append(pred_tokens)
                references.append([ref_tokens])

    return bleu_score(candidates, references, n_gram=4)


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
    tokenizer.add_tokens(['power','divide','sqrt','const0','const1','out0','out1'])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # 3) Dry-run
    dry_run(EncoderDecoderRNN, config)

    # 4) Data loading (with fallback)
    data_path = 'data/mathqa_train.json'
    if os.path.exists(data_path):
        ds = MathGraphDataset(
            path=data_path,
            tokenizer_src=tokenizer,
            tokenizer_tgt=tokenizer,
            max_src_len=config['max_src_len'],
            max_tgt_len=config['max_tgt_len']
        )
        print(f"Loaded local JSON ({data_path})")
    else:
        print("Local file not found, pulling HF 'math_qa' split and caching")
        raw = load_dataset('math_qa', 'all', split='train', trust_remote_code=True)
        os.makedirs('tests', exist_ok=True)
        tmp_path = 'tests/full_mathqa.json'
        with open(tmp_path, 'w') as f:
            json.dump([dict(ex) for ex in raw], f)
        ds = MathGraphDataset(
            path=tmp_path,
            tokenizer_src=tokenizer,
            tokenizer_tgt=tokenizer,
            max_src_len=config['max_src_len'],
            max_tgt_len=config['max_tgt_len']
        )
    loader = DataLoader(ds, batch_size=config['bs'], shuffle=True)
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epochs'], eta_min=1e-5)

    # 6) Training loop with best-model saving
    best_val_loss = float('inf')
    for epoch in range(1, config['max_epochs'] + 1):
        model.train()
        train_loss, train_correct, train_tokens = 0.0, 0, 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            rel_labels = batch['relation_labels'].to(device)

            optimizer.zero_grad()
            rel_logits, _ = model(input_ids, attention_mask)

            b, t, v = rel_logits.size()
            loss = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)(
                rel_logits.view(-1, v), rel_labels.view(-1)
            )
            loss.backward()
            optimizer.step()

            preds = rel_logits.argmax(dim=-1)
            mask = rel_labels != tokenizer.pad_token_id
            train_correct += (preds == rel_labels).masked_select(mask).sum().item()
            train_tokens  += mask.sum().item()
            train_loss    += loss.item()

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, tokenizer, device)
        val_bleu = compute_bleu(model, val_loader, tokenizer, device)

        # Scheduler
        scheduler.step()

        # Log metrics
        wandb.log({
            'Epoch': epoch,
            'Loss/train': train_loss / train_tokens,
            'Acc/train': train_correct / train_tokens,
            'Loss/val': val_loss,
            'Acc/val': val_acc,
            'BLEU/val': val_bleu,
            'LR': scheduler.get_last_lr()[0]
        })

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(wandb.run.dir, 'best_model.pt')
            torch.save(model.state_dict(), save_path)
            wandb.save(save_path)
            print(f"Saved new best model at epoch {epoch} (val_loss={val_loss:.4f})")

        print(f"Epoch {epoch}/{config['max_epochs']} — "
              f"Train Loss: {train_loss/train_tokens:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val BLEU: {val_bleu:.4f}")

    wandb.finish()


    