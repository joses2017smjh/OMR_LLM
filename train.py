import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torcheval.metrics.functional import bleu_score
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from transformers import AutoTokenizer
import wandb
from tqdm import tqdm

import time
from datetime import datetime

import random
import os
import argparse

from data.MathQA import MathQA
from models.DecoderTransformer import DecoderTransformer


# training configuration
train_config = {
    'bs': 32,
    'lr': 0.001,
    'weight_decay': 0.00001,
    'max_epochs': 30
}

# model configuration
model_config = {
    'emb_dim': 256,
    'num_layers': 8,
    'num_heads': 8
}


def dry_run(model, device, src_vocab_len, trg_vocab_len):

    B = train_config["bs"]
    src_len = random.randint(0, 100)
    trg_len = random.randint(0, 100)

    src_seq = torch.randint(0, src_vocab_len, (B, src_len)).to(device)
    trg_seq = torch.randint(0, trg_vocab_len, (B, trg_len)).to(device)
    out = model(src_seq, trg_seq)
  
    assert out.shape == (B, trg_len, trg_vocab_len)
    print("Passed dry run")


if __name__ == '__main__':
    
    now = datetime.now()
    run_name = "decoder_transformer_" + now.strftime("%Y_%m_%d_%H_%m")

    # initialize wandb session
    wandb.login()
    wandb.init(project="MathQA Decoder Transformer",name=run_name, config=train_config)

    # load MathQA train dataset
    train_set = MathQA(split='train')

    # get source and target vocabs
    src_vocab = train_set.src_vocab
    trg_vocab = train_set.trg_vocab

    # load MathQA validation dataset
    validation_set = MathQA(src_vocab=src_vocab, split='validation')

    # get source and target vocab lengths
    src_vocab_len = len(src_vocab)
    trg_vocab_len = len(trg_vocab)

    # create dataloaders
    train_loader =DataLoader(train_set, batch_size=train_config['bs'], num_workers=4, shuffle=True, collate_fn=train_set.pad_collate, drop_last=True)
    validation_loader =DataLoader(validation_set, batch_size=train_config['bs'], num_workers=4, shuffle=True, collate_fn=train_set.pad_collate, drop_last=True)

    # get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')

    # create model
    model = DecoderTransformer(
        src_vocab_size=src_vocab_len,
        trg_vocab_size=trg_vocab_len,
        embed_dim=model_config['emb_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads']
    ).to(device)

    # run through dummy data
    dry_run(model, device, src_vocab_len, trg_vocab_len)

    # set up optimizer with custom learning rate and weight decay
    optimizer = optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay']) 

    # define warmup and cooldown epochs
    warmup_epochs = int(train_config['max_epochs'] / 10)
    cooldown_epochs = train_config['max_epochs'] - warmup_epochs

    # epoch length
    epoch_len = len(train_loader)

    # construct linear warmup and cosine annealing scheduler
    linear = LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=warmup_epochs*epoch_len)
    cosine = CosineAnnealingLR(optimizer, T_max=cooldown_epochs*epoch_len, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs*epoch_len])

    # set up cross entropy loss for transformer output
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # set up progress bar
    pbar = tqdm(total=train_config['max_epochs']*epoch_len, desc="Training Iterations", unit="batch")

    # main training loop
    iteration = 0
    for epoch in range(train_config['max_epochs']):

        # set model to train
        model.train()

        for batch in train_loader:

            # log lr for each epoch
            wandb.log({'LR': scheduler.get_last_lr()[0]}, step=iteration)

            # get word problem and linear formula
            src_seq = batch[0].to(device)
            trg_seq = batch[1].to(device)

            # run through decoder-only transformer
            out = model(src_seq, trg_seq)[:,:-1,:]
            trg_seq = trg_seq[:,1:]

            # compute loss
            loss = criterion(out.permute(0,2,1), trg_seq)

            loss.backward()
            optimizer.step()

            # compute accuracy
            num_tokens = (trg_seq != 0).to(dtype=torch.float32).sum().item()
            accuracy = (torch.argmax(out, dim=2)==trg_seq).to(dtype=torch.float32).sum() / num_tokens

            optimizer.zero_grad()

            # log loss/train and accuracy/train per batch
            wandb.log({"Loss/train": loss.item()}, step=iteration)
            wandb.log({"Accuracy/train": accuracy}, step=iteration)
            
            pbar.update(1)
            iteration += 1

            # step through scheduler
            scheduler.step()
    
        # save model each epoch
        torch.save({
            'epoch':epoch,
            'model_state_dict': model.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_config': train_config,
            'model_config': model_config,
            'src_vocab': src_vocab,
            'trg_vocab': trg_vocab},
            "./chkpts/"+run_name+"_e"+str(epoch))
        
        # set model to evaluate
        model.eval()

        # validation loss
        val_loss = 0
        val_accuracy = 0

        for batch in validation_loader:

            # get word problem and linear formula
            src_seq = batch[0].to(device)
            trg_seq = batch[1].to(device)

            # run through decoder-only transformer
            out = model(src_seq, trg_seq)[:,:-1,:]
            trg_seq = trg_seq[:,1:]

            # compute loss
            val_loss += criterion(out.permute(0,2,1), trg_seq).item()

            # compute accuracy
            num_tokens = (trg_seq != 0).to(dtype=float).sum().item()
            val_accuracy += (torch.argmax(out, dim=2)==trg_seq).to(dtype=float).sum() / num_tokens

        # log loss/val and accuracy/val per batch
        wandb.log({"Loss/val": val_loss/len(validation_loader)}, step=iteration)
        wandb.log({"Accuracy/val": val_accuracy/len(validation_loader)}, step=iteration)

    wandb.finish()
    pbar.close()