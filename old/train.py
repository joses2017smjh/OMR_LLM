import os
import json
import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
import wandb
from torcheval.metrics.functional import bleu_score
from datasets import load_dataset
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm


import random

from dataloader import MathGraphDataset
from models.base import BaseModel
from models.rnn_model import EncoderDecoderRNN
#from models.rnn_model_no_tf import EncoderDecoderRNNnoTF

ACCUM_STEPS  = int(os.getenv("ACCUM_STEPS",  1))

init_tf_ratio = 1.0

# Configuration
config = {
    'project': 'MathQA-Graph-Translator-teacher_forcing_RNN',
    'bs': int(os.getenv("BATCH_SIZE", 32)),
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'embed_dim': 256,
    'hidden_dim': 512,
    'num_layers': 2,
    'max_epochs': 20,
    'max_src_len': 64,
    'max_tgt_len': 64

}



# Argument parsing for teacher-forcing vs normal RNN
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-teacher', action='store_true', help='Disable teacher forcing')
    return parser.parse_args()

def generate_strict(model, input_ids, attention_mask, max_steps):
    model.eval()
    B = input_ids.size(0)
    decoder = torch.full((B,1), SOS_ID, device=input_ids.device)
    outputs = [[] for _ in range(B)]
    with torch.no_grad():
        for t in range(max_steps):
            rlog, elog = model(input_ids, attention_mask, decoder)
            if t % 3 == 0:
                # mask out non-op positions
                logits = torch.full_like(rlog[:, -1], float('-inf'))
                logits[:, :relation_vocab_size] = rlog[:, -1][:, :relation_vocab_size]
            else:
                # mask out op‐positions, keep only full‐vocab
                logits = torch.full_like(elog[:, -1], float('-inf'))
                logits[:, relation_vocab_size:] = elog[:, -1][:, relation_vocab_size:]
            nxt = logits.argmax(dim=-1, keepdim=True)
            decoder = torch.cat([decoder, nxt], dim=1)
            for i in range(B):
                token_id = nxt[i].item()
                if token_id == tokenizer.eos_token_id:
                    # pad the rest so outputs[i] is length max_steps
                    outputs[i].extend([tokenizer.pad_token_id]*(max_steps - len(outputs[i])))
                    break
                outputs[i].append(token_id)
        return outputs




def dry_run(model_cls, config, relation_vocab_size, entity_vocab_size, device=None):
    """
    Quick sanity check: runs a batch of synthetic data through the model.
    """
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model_cls(
        src_vocab_size=len(tokenizer),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        relation_vocab_size = relation_vocab_size,
        entity_vocab_size   = entity_vocab_size
    ).to(device)

    batch_size = config['bs']
    src_len = config['max_src_len']
    tgt_len = config['max_tgt_len']

    # random source inputs
    input_ids = torch.randint(0, len(tokenizer), (batch_size, src_len), device=device)
    attention_mask = torch.ones_like(input_ids, device=device)
    decoder_input  = torch.full((batch_size, tgt_len), SOS_ID, dtype=torch.long, device=device)

    with torch.no_grad():
        rel_logits, ent_logits = model(input_ids, attention_mask, decoder_input)


    assert rel_logits.shape == (batch_size, tgt_len, relation_vocab_size)
    assert ent_logits.shape == (batch_size, tgt_len, entity_vocab_size)
    print(f"Dry run: rel={rel_logits.shape}, ent={ent_logits.shape}")

def evaluate(model, loader, device, use_teacher,
             rel_criterion, ent_criterion,
             relation_vocab_size, entity_vocab_size):
    model.eval()
    total_loss = 0.0
    total_rel_correct, total_rel_tokens = 0, 0
    total_ent_correct, total_ent_tokens = 0, 0
    with torch.no_grad():
        for batch in loader:
            ids      = batch['input_ids'].to(device)
            att      = batch['attention_mask'].to(device)
            full_lbl = batch['full_labels'].to(device)
            rel_lbl  = batch['relation_labels'].to(device)
            ent_lbl  = batch['entity_labels'].to(device)

            if use_teacher:
                decoder_in = torch.full_like(full_lbl, SOS_ID)
                decoder_in[:,1:] = full_lbl[:,:-1]
                rel_logits, ent_logits = model(ids, att, decoder_in)
                
            else:
                rel_logits, ent_logits  = model(ids, att)

             # losses
            rl = rel_criterion(
                rel_logits.view(-1, relation_vocab_size),
                rel_lbl.view(-1)
            )
            el = ent_criterion(
                ent_logits.view(-1, entity_vocab_size),
                ent_lbl.view(-1)
            )
            total_loss += (rl + el).item()

            # accuracy
            rel_preds = rel_logits.argmax(dim=-1)
            ent_preds = ent_logits.argmax(dim=-1)

            rel_mask = rel_lbl != -100
            ent_mask = ent_lbl != tokenizer.pad_token_id

            total_rel_correct += rel_preds.masked_select(rel_mask).eq(rel_lbl.masked_select(rel_mask)).sum().item()
            total_rel_tokens  += rel_mask.sum().item()

            total_ent_correct += ent_preds.masked_select(ent_mask).eq(ent_lbl.masked_select(ent_mask)).sum().item()
            total_ent_tokens  += ent_mask.sum().item()

    avg_loss = total_loss / (total_rel_tokens + total_ent_tokens)
    rel_acc  = total_rel_correct / total_rel_tokens
    ent_acc  = total_ent_correct / total_ent_tokens
    
    total_correct = total_rel_correct + total_ent_correct
    total_tokens  = total_rel_tokens  + total_ent_tokens
    overall_acc = total_correct / total_tokens
    return avg_loss, (rel_acc, ent_acc, overall_acc)

def strip_specials(seq, sos_id, eos_id, pad_id):
    # drop leading SOS
    if seq and seq[0]==sos_id:
        seq = seq[1:]
    # truncate at first EOS or PAD
    for stop_id in (eos_id, pad_id):
        if stop_id in seq:
            seq = seq[:seq.index(stop_id)]
    return seq

def compute_bleu(model, loader, device):
    hyps, refs = [], []
    with torch.no_grad():
        for batch in loader:
            ids, att = batch['input_ids'].to(device), batch['attention_mask'].to(device)
            gold = batch['full_labels'].tolist()
            preds = generate_strict(model, ids, att, max_steps=len(gold[0]))

            for g_seq, p_seq in zip(gold, preds):
                g_seq = strip_specials(g_seq, SOS_ID,
                                       tokenizer.eos_token_id,
                                       tokenizer.pad_token_id)
                p_seq = strip_specials(p_seq, SOS_ID,
                                       tokenizer.eos_token_id,
                                       tokenizer.pad_token_id)
                refs.append([ tokenizer.decode(g_seq, skip_special_tokens=True) ])
                hyps.append( tokenizer.decode(p_seq, skip_special_tokens=True) )
    try:
        return bleu_score(hyps, refs, n_gram=4)
    except ValueError:
        return bleu_score(hyps, refs, n_gram=1)



if __name__ == '__main__':
    cpus = int(os.getenv('SLURM_CPUS_PER_TASK', 2))
    print(f"Using {cpus} DataLoader workers")
    
    args = parse_args()
    use_teacher = not args.no_teacher
    
    run_suffix = 'RNN_TF' if use_teacher else 'RNN_No_TF'
    run_name = f"{config['project']}_{run_suffix}"

    # 1) Initialize W&B
    wandb.login()
    wandb.init(project=config['project'],name=run_name, config=config)

    # 2) Tokenizer setup
    tokenizer = AutoTokenizer.from_pretrained(
        'meta-llama/Llama-3.2-1B',
        trust_remote_code=True,
        token=True
    )
    # ensure placeholders never get split
    # read the same two files
    with open('./data/MathQA/operation_list.txt') as f:
        operator_list = [l.strip() for l in f if l.strip()]
    with open('./data/MathQA/constant_list.txt') as f:
        constant_list = [l.strip() for l in f if l.strip()]


    # placeholders
    vars_  = [f'n{i}' for i in range(20)]
    hashes = [f'#{i}' for i in range(20)]

    all_special = operator_list + constant_list + vars_ + hashes
    tokenizer.add_special_tokens({
    'pad_token': '<PAD>',
    'bos_token': '<SOS>',
    'eos_token': '<EOS>',
    'additional_special_tokens': all_special
    })
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('<PAD>')
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('<EOS>')
    SOS_ID = tokenizer.bos_token_id

    # rebuild your small‐vocab map from file
    op2small_id = { op:i for i,op in enumerate(operator_list) }
    relation_vocab_size = len(operator_list)
    entity_vocab_size   = len(tokenizer)

        
    model_cls = EncoderDecoderRNN if use_teacher else EncoderDecoderRNNnoTF
    

    # 3) Dry-run
    dry_run(model_cls, config, relation_vocab_size, entity_vocab_size)

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
    train_loader = DataLoader(
    Subset(ds, list(range(val_split, len(ds)))),
    batch_size=config['bs'], shuffle=True,
    num_workers=cpus,
    pin_memory=True,
)

    val_loader = DataLoader(
    Subset(ds, list(range(val_split))),
    batch_size=config['bs'], shuffle=False,
    num_workers=max(1, cpus//2),
    pin_memory=True,
)

    # 5) Model, optimizer, scheduler
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_cls(
        src_vocab_size      = len(tokenizer),
        embed_dim          = config['embed_dim'],
        hidden_dim         = config['hidden_dim'],
        num_layers         = config['num_layers'],
        relation_vocab_size = relation_vocab_size,  # small head
        entity_vocab_size   = entity_vocab_size     # full head
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
    rel_criterion = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    ent_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id,
                                        reduction='sum')
    # Training loop with teacher forcing and step-level logs
    num_steps = config['max_epochs'] * len(train_loader)
    pbar = tqdm(total=num_steps, desc='Training Iterations', unit='batch')
    best_val_loss = float('inf')
    optimizer.zero_grad()
    iteration = 0

    for epoch in range(1, config['max_epochs'] + 1):
        tf_ratio = init_tf_ratio * (1 - (epoch-1)/(config['max_epochs']-1))  

        model.train()
        # log LR at start of epoch
        wandb.log({'LR': scheduler.get_last_lr()[0]}, step=iteration)

        for batch in train_loader:
            # 1) unpack batch
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            full_labels    = batch['full_labels'].to(device)    # full‐vocab targets
            rel_labels     = batch['relation_labels'].to(device) # small‐vocab (ops) targets
            ent_labels     = batch['entity_labels'].to(device)   # full‐vocab (args) targets

            # 2) build decoder inputs if teacher forcing
            B, T = full_labels.size()
            decoder_input = torch.full((B, T), SOS_ID, device=device)

            if use_teacher or random.random() < tf_ratio:
                # classic teacher forcing
                decoder_input[:,1:] = full_labels[:,:-1]
            else:
                # use a bit of the model’s own preds
                decoder_input[:,1] = full_labels[:,0]  # first real token = gold
                for t in range(1, T-1):
                    rel_logits, ent_logits = model(input_ids, attention_mask, decoder_input[:,:t+1])
                    if t % 3 == 0:
                        nxt = rel_logits[:, -1].argmax(-1)
                    else:
                        nxt = ent_logits[:, -1].argmax(-1)
                    decoder_input[:, t+1] = nxt

            rel_logits, ent_logits = model(input_ids, attention_mask, decoder_input)

            # 3) compute losses
            #   - rel_logits: (B, T, |ops|) vs rel_labels (B, T)
            #   - ent_logits: (B, T, V_full) vs ent_labels (B, T)
            rel_loss = rel_criterion(
                rel_logits.view(-1, relation_vocab_size),
                rel_labels.view(-1)
            )
            ent_loss = ent_criterion(
                ent_logits.view(-1, entity_vocab_size),
                ent_labels.view(-1)
            )
            loss = rel_loss + ent_loss

            # 4) backward & step
            (loss / ACCUM_STEPS).backward()
            iteration += 1
            if iteration % ACCUM_STEPS == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # 5) compute and log step-level accuracy
            with torch.no_grad():
                rel_preds = rel_logits.argmax(dim=-1)
                ent_preds = ent_logits.argmax(dim=-1)

                rel_mask = rel_labels != -100
                ent_mask = ent_labels != tokenizer.pad_token_id

                rel_acc = (rel_preds.masked_select(rel_mask)
                                .eq(rel_labels.masked_select(rel_mask))
                                .sum() / rel_mask.sum())
                ent_acc = (ent_preds.masked_select(ent_mask)
                                .eq(ent_labels.masked_select(ent_mask))
                                .sum() / ent_mask.sum())

            wandb.log({
                'Loss/train':   (rel_loss+ent_loss).item() / (rel_mask.sum()+ent_mask.sum()).item(),
                'Loss_rel/train': rel_loss.item() / rel_mask.sum().item(),
                'Loss_ent/train': ent_loss.item() / ent_mask.sum().item(),
                'Acc_rel/train':  rel_acc.item(),
                'Acc_ent/train':  ent_acc.item(),
            }, step=iteration)

            pbar.update(1)

        # ─── end of epoch: validation ───
        model.eval()
        val_loss, (rel_val_acc, ent_val_acc, val_acc) = evaluate(
            model, val_loader, device, use_teacher,
            rel_criterion, ent_criterion,
            relation_vocab_size, entity_vocab_size)
    

        val_bleu = compute_bleu(model, val_loader, device)

        wandb.log({
            'Loss/val': val_loss,
            'Acc_rel/val':  rel_val_acc,
            'Acc_ent/val':  ent_val_acc,
            'Acc/val':  val_acc,
            'BLEU/val': val_bleu,
            'Epoch':    epoch,
        }, step=iteration)
        print(f"Finished epoch {epoch} — val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(wandb.run.dir, 'best_model.pt')
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best model at epoch {epoch} (val_loss={val_loss:.4f})")

    wandb.finish()
    pbar.close()
