import torch
from torch import nn
from models.base import BaseModel

class EncoderDecoderRNN(BaseModel):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim, num_layers,
                 relation_vocab_size, entity_vocab_size):
        super().__init__(hidden_dim*2, relation_vocab_size, entity_vocab_size)
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        # decoder now takes embed_dim inputs, hidden size = hidden_dim*2
        self.decoder = nn.LSTM(embed_dim, hidden_dim*2, num_layers, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim*2, num_heads=8)
        # heads already defined in BaseModel → relation_head, entity_head

    def forward(self, input_ids, attention_mask, decoder_input_ids=None):
            # 1) Encode source sequence
            src_emb = self.src_embed(input_ids)               # (B, S, E)
            enc_out, (h_n, c_n) = self.encoder(src_emb)       # enc_out: (B, S, H*2)

            # 2) Prepare initial hidden/cell for decoder by merging bidirectional
            # h_n, c_n shape: (num_layers*2, B, H)
            num_layers = self.encoder.num_layers
            B = h_n.size(1)
            H = h_n.size(2)
            # reshape to (num_layers, 2, B, H)
            h_n = h_n.view(num_layers, 2, B, H)
            c_n = c_n.view(num_layers, 2, B, H)
            # concatenate forward/backward states
            h_dec = torch.cat([h_n[:, 0], h_n[:, 1]], dim=-1)  # (num_layers, B, 2H)
            c_dec = torch.cat([c_n[:, 0], c_n[:, 1]], dim=-1)

            # 3) Decode (teacher-forcing if decoder_input_ids provided)
            if decoder_input_ids is None:
                raise NotImplementedError("Inference mode not implemented yet; pass decoder_input_ids for training.")

            # embed decoder inputs
            dec_emb = self.src_embed(decoder_input_ids)       # (B, T, E)
            dec_out, _ = self.decoder(dec_emb, (h_dec, c_dec)) # (B, T, H*2)

            # 4) Attend: treat encoder output as keys+values
            # transpose to (T, B, H*2) for attention
            q = dec_out.transpose(0, 1)
            k = v = enc_out.transpose(0, 1)
            attn_out, _ = self.attn(q, k, v)
            dec_final = attn_out.transpose(0, 1)              # (B, T, H*2)

            # 5) Project to relation and entity logits
            rel_logits = self.relation_head(dec_final)        # (B, T, V)
            ent_logits = self.entity_head(dec_final)          # (B, T, V)
            return rel_logits, ent_logits



''''
ENCODER-DECODER RNN MODEL WITHOUT TEACHER FORCING


class EncoderDecoderRNN(BaseModel):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim, num_layers, relation_vocab_size, entity_vocab_size):
        super().__init__(hidden_dim*2, relation_vocab_size, entity_vocab_size)
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim)
        self.encoder = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.LSTM(embed_dim, hidden_dim*2, num_layers, batch_first=True)
        self.attn = nn.MultiheadAttention(hidden_dim*2, num_heads=8)

    def forward(self, input_ids, attention_mask, labels=None):
        src_emb = self.src_embed(input_ids)
        enc_out, _ = self.encoder(src_emb)
        # placeholder decoder logic
        dec_hidden = enc_out

        rel_logits = self.relation_head(dec_hidden)
        ent_logits = self.entity_head(dec_hidden)
        return rel_logits, ent_logits
'''