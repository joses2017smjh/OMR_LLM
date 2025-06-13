import torch
from torch import nn
import math


# get device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.embed_dim = embed_dim

    def forward(self, x):

        # batch size and sequence length bookkeeping
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # initialize positional encoding
        pe = torch.zeros(1, seq_len, self.embed_dim).to(x.device)

        # calculate encoding term
        pos = torch.arange(0, seq_len, dtype=torch.float)
        enc = torch.exp((-math.log(10000.0)) * (torch.arange(0, self.embed_dim, step=2, dtype=torch.float) / self.embed_dim))

        # calculate positional encoding
        prod = torch.outer(pos, enc)
        pe[0, :, 0::2] = torch.sin(prod)
        pe[0, :, 1::2] = torch.cos(prod)
        pe = pe.expand(batch_size, -1, -1)

        # apply as residual
        x = x + pe
        return x


class DecoderTransformer(nn.Module):

    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            embed_dim,
            num_layers,
            num_heads
    ):
        super().__init__()

        # learned vector embeddings for source and target sequences
        self.src_token_embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=embed_dim)
        self.trg_token_embedding = nn.Embedding(num_embeddings=src_vocab_size, embedding_dim=embed_dim)

        # apply separate positional encodings
        self.pos_enc = PositionalEncoding(embed_dim=embed_dim)

        # prepare single transformer layer with multiheaded attention
        # name is a bit of a misnomer, we use encoder layers as vanilla transformer layers
        transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)

        # create decoder with multiple layers
        # name is, again, a bit of a misnomer
        self.decoder = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_layers)

        # classifier (only needed for target vocab)
        self.classifier = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=trg_vocab_size)
        )

    
    def forward(self, src_seq, trg_seq):

        # get lengths of sequences
        src_len = src_seq.shape[1]
        trg_len = trg_seq.shape[1]

        # embed sequences through learned vector embeddings
        src_seq_embed = self.src_token_embedding(src_seq)
        trg_seq_embed = self.trg_token_embedding(trg_seq)

        # add positional embeddings to vector embeddings
        src_seq_embed = self.pos_enc(src_seq_embed)
        trg_seq_embed = self.pos_enc(trg_seq_embed)

        # concatenate sequences together
        seq = torch.cat((src_seq_embed, trg_seq_embed), dim=1)

        # generate custom prefixed causal mask
        src_mask = torch.zeros(src_len+trg_len, src_len, dtype=torch.float32)
        
        trg_mask_1 = torch.ones(src_len, trg_len, dtype=torch.float32) * float('-inf')
        trg_mask_2 = torch.ones(trg_len, trg_len, dtype=torch.float32) * float('-inf')
        trg_mask_2 = torch.tril(trg_mask_2, diagonal=-1).T
        trg_mask = torch.cat((trg_mask_1, trg_mask_2), dim=0)

        causal_mask = torch.cat((src_mask, trg_mask), dim=1).to(device)

        # run sequence through decoder
        seq_out = self.decoder(src=seq, mask=causal_mask)

        # decouple source and target sequence outputs
        _, trg_seq_out = torch.split(seq_out, [src_len, trg_len], dim=1)

        # classify target sequence output into target vocabulary
        out = self.classifier(trg_seq_out)
        return out