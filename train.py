import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torcheval.metrics.functional import bleu_score
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from transformers import AutoTokenizer
import wandb
from tqdm import tqdm

import random
import os
import argparse

from data.MathQA import MathQA
from models.EncoderDecoderRNN import EncoderDecoderRNN


ACCUM_STEPS  = int(os.getenv("ACCUM_STEPS",  1))

init_tf_ratio = 1.0

# training configuration
config = {
    'model': 'rnn',
    'bs': 32,
    'lr': 1e-3,
    'weight_decay': 1e-5,
    'embed_dim': 256,
    'hidden_dim': 512,
    'num_layers': 2,
    'max_epochs': 20,
    'max_src_len': 128,
    'max_tgt_len': 128
}


def dry_run(model, device):

    B = config["bs"]
    L_src = 100
    L_trg = 200
    V_src = 1000
    V_trg = 2000

    src = torch.randint(0, V_src, (B, L_src)).to(device)
    trg = torch.randint(0, V_trg, (B, L_trg)).to(device)
    out = model(src, trg)

    print(out.shape())

    #loss = out.sum()
    #loss.backward()
    
    #assert out.shape == (B, L, V), "[FAILED] dry fit to check shapes work for a dummy input."
    #print("[Passed] Dry fit to check shapes work for a dummy input.")






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

    run_name = "rnn"

    # initialize W&B
    # wandb.login()
    # wandb.init(project=config['project'],name=run_name, config=config)

    # load MathQA dataset
    mathqa = MathQA()
    mathqa_len = mathqa.__len__()

    train_percent = 0.8
    train_len = int(mathqa_len*0.8)
    test_len = mathqa_len - train_len

    # split into training and testing datasets
    train_set, test_set = torch.utils.data.random_split(mathqa, [train_len, test_len])

    # create dataloaders
    trainloader =DataLoader(train_set, batch_size=4, num_workers=4, shuffle=True, collate_fn=mathqa.pad_collate, drop_last=True)
    testloader = DataLoader(test_set, batch_size=4, num_workers=4, shuffle=True, collate_fn=mathqa.pad_collate, drop_last=True)

    # get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')

    # create model
    model = EncoderDecoderRNN(
        src_vocab_size=mathqa.src_vocab.__len__(),
        trg_vocab_size=mathqa.trg_vocab.__len__(),
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
    ).to(device)

    dry_run(model, device)




    # 5) Model, optimizer, scheduler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_cls = EncoderDecoderRNN
    model = model_cls(
        src_vocab_size=...,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        relation_vocab_size=relation_vocab_size,  # small head
        entity_vocab_size=entity_vocab_size     # full head
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