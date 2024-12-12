import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

# Define the AddPositionalEncoding class with standard sinusoidal encoding
class AddPositionalEncoding(nn.Module):
    def __init__(self, len_max):
        super().__init__()
        self.len_max = len_max

    def forward(self, input):
        # input shape: (batch_size, seq_length, dim_model)
        batch_size, seq_length, dim_model = input.size()
        position = torch.arange(seq_length, device=input.device).unsqueeze(1)  # (seq_length, 1)
        div_term = torch.exp(torch.arange(0, dim_model, 2, device=input.device) * -(math.log(10000.0) / dim_model))
        pe = torch.zeros(seq_length, dim_model, device=input.device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_length, dim_model)
        return input + pe

# Define the QKVAttention class
class QKVAttention(nn.Module):
    def __init__(self, dim_in, dim_qk, dim_v, n_heads=1, causal=False, dropout=0.0):
        super().__init__()

        def randw(*d):
            return nn.Parameter(torch.randn(*d) / math.sqrt(d[-1]))

        self.causal = causal
        self.dropout = dropout

        self.w_q = randw(n_heads, dim_qk, dim_in)
        self.w_k = randw(n_heads, dim_qk, dim_in)
        self.w_v = randw(n_heads, dim_v, dim_in)
        self.w_o = randw(dim_v * n_heads, dim_in)

    def forward(self, input):
        # input shape: (batch_size, seq_length, dim_in)
        q = torch.einsum("ntc,hdc->nhtd", input, self.w_q)  # (batch_size, nb_heads, seq_length, dim_qk)
        k = torch.einsum("ntc,hdc->nhtd", input, self.w_k)  # (batch_size, nb_heads, seq_length, dim_qk)
        v = torch.einsum("ntc,hdc->nhtd", input, self.w_v)  # (batch_size, nb_heads, seq_length, dim_v)

        a = torch.einsum("nhtd,nhsd->nhts", q, k) / math.sqrt(self.w_q.size(1))  # (batch_size, nb_heads, seq_length, seq_length)

        if self.causal:
            t = torch.arange(input.size(1), device=q.device)
            attzero = t[None, None, :, None] < t[None, None, None, :]
            a = a.masked_fill(attzero, float("-inf"))

        a = a.softmax(dim=3)  # Softmax over the last dimension
        a = F.dropout(a, self.dropout, self.training)
        y = torch.einsum("nhts,nhsd->nthd", a, v).flatten(2)  # (batch_size, seq_length, nb_heads * dim_v)

        return y

# Define the TransformerBlock class
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, dim_keys, dim_hidden, n_heads, causal, dropout):
        super().__init__()
        self.att_ln = nn.LayerNorm(n_embd)
        self.att_mh = QKVAttention(
            dim_in=n_embd,
            dim_qk=dim_keys,
            dim_v=n_embd // n_heads,
            n_heads=n_heads,
            causal=causal,
            dropout=dropout,
        )
        self.ffn_ln = nn.LayerNorm(n_embd)
        # self.ffn_fc1 = nn.Linear(in_features=dim_model, out_features=dim_hidden)
        # self.ffn_fc2 = nn.Linear(in_features=dim_hidden, out_features=dim_model)
        # self.gelu_activation = NewGELU()
        self.mlp = nn.ModuleDict(dict(
            ffn_fc1 = nn.Linear(n_embd, dim_hidden),
            ffn_fc2 = nn.Linear(dim_hidden, n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(0.1)
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.ffn_fc2(m.act(m.ffn_fc1(x)))) # MLP forward

    def forward(self, x):
        x = x + self.att_mh(self.att_ln(x))
        x = x + self.mlpf(self.ffn_ln(x))
        return x

# Define the PicoGPT model
class PicoGPT(nn.Module):
    def __init__(
        self,
        voc_size,
        n_embd,
        dim_keys,
        dim_hidden,
        n_heads,
        nb_blocks,
        causal,
        dropout=0.0,
        len_max=1e5,
    ):
        super().__init__()

        self.starter = nn.Sequential(
            nn.Embedding(voc_size, n_embd, padding_idx=tokenizer.pad_token_id),
            nn.Dropout(dropout),
            AddPositionalEncoding(len_max),
        )

        self.trunk = nn.Sequential(
            *[
                TransformerBlock(
                    n_embd, dim_keys, dim_hidden, n_heads, causal, dropout
                )
                for _ in range(nb_blocks)
            ]
        )

        self.readout = nn.Linear(in_features=n_embd, out_features=voc_size)

        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Embedding):
                    m.weight.normal_(mean=0, std=2e-2)
                elif isinstance(m, nn.LayerNorm):
                    m.bias.zero_()
                    m.weight.fill_(1.0)

    def forward(self, input):
        # input shape: (batch_size, seq_length)
        x = self.starter(input)  # (batch_size, seq_length, dim_model)
        x = self.trunk(x)        # (batch_size, seq_length, dim_model)
        x = self.readout(x)      # (batch_size, seq_length, voc_size)
        return x

    def cross_entropy(self, input):
        x = self(input)
        return F.cross_entropy(x.transpose(1, 2), input)