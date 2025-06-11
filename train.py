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
from models.DecoderTransformer import DecoderTransformer


# training configuration
train_config = {
    'bs': 32,
    'lr': 0.001,
    'weight_decay': 0.00001,
    'max_epochs': 10
}

# model configuration
model_config = {
    'emb_dim': 256,
    'num_layers': 2,
    'num_heads': 8
}


def dry_run(model, device, trg_vocab_len):

    B = train_config["bs"]
    src_len = 100
    trg_len = 50

    src_seq = torch.randint(0, src_vocab_len, (B, src_len)).to(device)
    trg_seq = torch.randint(0, trg_vocab_len, (B, trg_len)).to(device)
    out = model(src_seq, trg_seq)
  
    assert out.shape == (B, trg_len, trg_vocab_len)
    print("Passed dry run")


if __name__ == '__main__':
    
    run_name = "decoder_transformer_test"

    # initialize wandb session
    wandb.login()
    wandb.init(project="decoder_transformer",name=run_name, config=train_config)

    # load MathQA dataset
    mathqa = MathQA()

    # get lengths
    mathqa_len = mathqa.__len__()
    src_vocab_len = mathqa.src_vocab.__len__()
    trg_vocab_len = mathqa.trg_vocab.__len__()

    # split into training and testing datasets
    train_percent = 0.8
    train_len = int(mathqa_len*train_percent)
    test_len = mathqa_len - train_len

    train_set, test_set = torch.utils.data.random_split(mathqa, [train_len, test_len])

    # create dataloaders
    trainloader =DataLoader(train_set, batch_size=train_config['bs'], num_workers=4, shuffle=True, collate_fn=mathqa.pad_collate, drop_last=True)
    testloader = DataLoader(test_set, batch_size=train_config['bs'], num_workers=4, shuffle=True, collate_fn=mathqa.pad_collate, drop_last=True)

    # get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')

    # create model
    model = DecoderTransformer(
        src_vocab_size=src_vocab_len,
        trg_vocab_size=trg_vocab_len,
        embed_dim=model_config['emb_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        max_src_len=100
    ).to(device)

    # run through dummy data
    dry_run(model, device, trg_vocab_len)

    # set up optimizer with custom learning rate and weight decay
    optimizer = optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay']) 

    # define warmup and cooldown epochs
    warmup_epochs = int(train_config['max_epochs'] / 10)
    cooldown_epochs = train_config['max_epochs'] - warmup_epochs

    # get number of iterations per epoch
    epoch_len = trainloader.__len__()

    # construct linear warmup and cosine annealing scheduler
    linear = LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=warmup_epochs*epoch_len)
    cosine = CosineAnnealingLR(optimizer, T_max=cooldown_epochs*epoch_len, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs*epoch_len])

    # set up cross entropy loss for transformer output
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # set up progress bar
    pbar = tqdm(total=train_config["max_epochs"]*train_len, desc="Training Iterations", unit="batch")

    # main training loop
    iteration = 0
    for epoch in range(train_config['max_epochs']):

        # set model to train
        model.train()

        # log lr for each epoch
        wandb.log({'LR': scheduler.get_last_lr()[0]}, step=iteration)

        for batch in trainloader:

            # get word problem and linear formula
            src_seq = batch[0].to(device)
            trg_seq = batch[1].to(device)

            # run through decoder-only transformer
            out = model(src_seq, trg_seq)[:,:-1,:]
            trg_seq = trg_seq[:,1:]

            loss = criterion(out.permute(0,2,1), trg_seq)

            loss.backward()
            optimizer.step()

            optimizer.zero_grad()

            # log loss/train per batch
            wandb.log({"Loss/train": loss.item()}, step=iteration)
            
            pbar.update(1)
            iteration += 1
    
        torch.save({
            'epoch':epoch,
            'model_state_dict': model.state_dict()},
            "./chkpts/"+run_name+"_"+str(epoch))

        # step through scheduler
        scheduler.step()

    wandb.finish()
    pbar.close()