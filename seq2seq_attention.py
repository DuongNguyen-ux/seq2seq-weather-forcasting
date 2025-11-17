# seq2seq_attention.py
import torch
import torch.nn as nn

# ===== Luong Attention (general) =====
class LuongAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        # general score: h_t^T W h_s
        self.attn = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [num_layers, batch, hidden_dim] -> use last layer
        # encoder_outputs: [batch, src_len, hidden_dim]
        h = hidden[-1]  # [batch, hidden_dim]
        # (batch, 1, hidden_dim) x (batch, hidden_dim, src_len) -> (batch,1,src_len)
        energy = torch.bmm(self.attn(h).unsqueeze(1), encoder_outputs.transpose(1, 2))
        attn_weights = torch.softmax(energy, dim=-1)  # [batch, 1, src_len]
        context = torch.bmm(attn_weights, encoder_outputs)  # [batch,1,hidden_dim]
        context = context.squeeze(1)  # [batch, hidden_dim]
        attn_weights = attn_weights.squeeze(1)  # [batch, src_len]
        return context, attn_weights


# ===== Encoder =====
class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, src):
        # src: [batch, src_len, input_dim]
        outputs, (hidden, cell) = self.lstm(src)
        # outputs: [batch, src_len, hidden_dim]
        return outputs, hidden, cell


# ===== Decoder with Attention =====
class DecoderLSTMWithAttn(nn.Module):
    def __init__(self, output_dim, hidden_dim, num_layers=1):
        super().__init__()
        # input to LSTM will be prev_y concatenated with context vector
        self.lstm = nn.LSTM(output_dim + hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.attn = LuongAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, prev_y, hidden, cell, encoder_outputs):
        # prev_y: [batch, 1, output_dim]
        # hidden: [num_layers, batch, hidden_dim]
        context, attn_weights = self.attn(hidden, encoder_outputs)  # context: [batch, hidden_dim]
        context = context.unsqueeze(1)  # [batch,1,hidden_dim]
        lstm_input = torch.cat([prev_y, context], dim=-1)  # [batch,1, output_dim+hidden_dim]
        out, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        pred = self.fc(out)  # [batch,1,output_dim]
        return pred, hidden, cell, attn_weights


# ===== Full Seq2Seq with Attention =====
class Seq2SeqAttn(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1):
        super().__init__()
        self.encoder = EncoderLSTM(input_dim, hidden_dim, num_layers)
        self.decoder = DecoderLSTMWithAttn(output_dim, hidden_dim, num_layers)

    def forward(self, src, target_len, teacher_forcing_ratio=0.0, y_true=None):
        # src: [batch, src_len, input_dim]
        encoder_outputs, hidden, cell = self.encoder(src)
        batch_size = src.size(0)
        device = src.device
        outputs = []
        prev_y = torch.zeros(batch_size, 1, 1).to(device)  # initial input
        for t in range(target_len):
            pred, hidden, cell, attn_w = self.decoder(prev_y, hidden, cell, encoder_outputs)
            outputs.append(pred)
            # teacher forcing
            if (y_true is not None) and (torch.rand(1).item() < teacher_forcing_ratio):
                # y_true assumed shape [batch, target_len] or [batch, target_len, 1]
                if y_true.dim() == 2:
                    prev_y = y_true[:, t:t+1].unsqueeze(-1).to(device)
                else:
                    prev_y = y_true[:, t:t+1].to(device)
            else:
                prev_y = pred.detach()
        outputs = torch.cat(outputs, dim=1)  # [batch, target_len, output_dim]
        return outputs
