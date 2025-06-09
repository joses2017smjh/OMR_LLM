import torch
from torch import nn

class EncoderDecoderRNN(nn.Module):

    def __init__(self, src_vocab_size, trg_vocab_size, embed_dim, hidden_dim, num_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.src_vocab_size = src_vocab_size
        self.trg_vocab_size = trg_vocab_size
        
        # encoder LMST
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        
        # decoder LSTM
        self.decoder = nn.LSTM(embed_dim, hidden_dim*2, num_layers, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim*2, num_heads=8)

        # classification head
        self.head = nn.Linear(hidden_dim, trg_vocab_size)


    def forward(self, src, trg):
        # decoder outputs
        outputs = torch.zeros(trg.shape[0], trg.shape[1], self.trg_vocab_size).to(src.device)

        # embed source sequence
        src_emb = self.src_embed(src)

        # run through encoder
        enc_out, _ = self.encoder(src_emb)

        # one-hot encode <SOS> character for each batch element
        outputs[:,0, 1] = 1

        # perform teacher-forcing on decoder
        for t in range(1, trg.shape[1]):
            outputs[:,t], _ = self.decoder(trg[:,t-1])

        return outputs


    """
    def evaluate(self, src, src_lens, max_len):
        # decoder outputs
        outputs = torch.zeros(src.shape[0], max_len).to(src.device)

        # get <SOS> inputs
        input_words = torch.ones(src.shape[0], dtype=torch.long, device=src.device)

        # embed source sequence
        src_emb = self.src_embed(src)

        # run source sentence through encoder, format properly
        enc_out, (h_n, c_n) = self.encoder(src_emb)

        # auto-regress to generate translated sentence
        for t in range(max_len):
            hidden, output = self.decoder(input_words)
            input_words = torch.argmax(output, dim=1)
            outputs[:,t] = input_words

        return outputs
    """