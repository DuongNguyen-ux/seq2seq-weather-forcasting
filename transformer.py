# transformer.py
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return x


class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_encoder_layers=2,
                 num_decoder_layers=2, dim_feedforward=128, dropout=0.1, target_len=24):
        super().__init__()
        self.d_model = d_model
        self.input_dim = input_dim
        self.target_len = target_len

        self.src_proj = nn.Linear(input_dim, d_model)
        self.tgt_proj = nn.Linear(1, d_model)

        self.pos_enc = PositionalEncoding(d_model, max_len=500)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)
        self.out_fc = nn.Linear(d_model, 1)

    def forward(self, src, target_len=None):
        # src: [batch, src_len, input_dim]
        batch_size, src_len, _ = src.size()
        if target_len is None:
            target_len = self.target_len

        src_emb = self.src_proj(src)  # [batch, src_len, d_model]
        src_emb = self.pos_enc(src_emb)

        # build initial decoder input (zeros)
        tgt = torch.zeros(batch_size, target_len, 1, device=src.device)
        tgt_emb = self.tgt_proj(tgt)  # [batch, target_len, d_model]
        tgt_emb = self.pos_enc(tgt_emb)

        # transformer expects src and tgt (batch_first=True)
        out = self.transformer(src_emb, tgt_emb)  # [batch, target_len, d_model]
        out = self.out_fc(out)  # [batch, target_len, 1]
        return out.squeeze(-1)  # [batch, target_len]
