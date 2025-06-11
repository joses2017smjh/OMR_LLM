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


chkpt_path = "./chkpts/decoder_transformer_test_9"
chkpt = torch.load(chkpt_path, weights_only=False, map_location=torch.device('mps'))

decoder = DecoderTransformer(
    src_vocab_size=mathqa.src_vocab.__len__(),
    trg_vocab_size=mathqa.trg_vocab.__len__(),
    embed_dim=256,
    num_layers=12,
    num_heads=8,
    max_trg_len=100,
)

#decoder.load_state_dict(chkpt['model_state_dict'])
decoder.to('mps')

trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
print(trainable_params)