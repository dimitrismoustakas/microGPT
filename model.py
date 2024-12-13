import math
import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F

# QKVAttention class with optional causality
class QKVAttention(nn.Module):
    def __init__(self, n_embd, n_heads=1, block_size = 1024, dropout=0.0, use_bias = False, causal = False):
        super().__init__()

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias = use_bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=use_bias)

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_heads
        self.n_embd = n_embd
        self.dropout = dropout
        self.causal = causal

        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            if self.causal:
                self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))
            else:
                self.register_buffer("bias", torch.ones(block_size, block_size)
                                 .view(1, 1, block_size, block_size))
            
    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=self.causal)
        else:
            if self.causal:
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        
        return y

# TransformerBlock class
class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_heads, block_size, dropout, use_bias, causal):
        super().__init__()
        self.att_ln = nn.LayerNorm(n_embd, bias = use_bias)
        self.att_mh = QKVAttention(
            n_embd=n_embd,
            n_heads=n_heads,
            block_size=block_size,
            dropout=dropout,
            use_bias=use_bias,
            causal=causal,
        )
        self.ffn_ln = nn.LayerNorm(n_embd, bias = use_bias)
        self.mlp = nn.ModuleDict(dict(
            ffn_fc1 = nn.Linear(n_embd, 4*n_embd),
            ffn_fc2 = nn.Linear(4*n_embd, n_embd),
            act     = nn.GELU(),
            dropout = nn.Dropout(0.1)
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.ffn_fc2(m.act(m.ffn_fc1(x)))) # MLP forward

    def forward(self, x):
        x = x + self.att_mh(self.att_ln(x))
        x = x + self.mlpf(self.ffn_ln(x))
        return x

# PicoGPT model
class PicoGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd,
        n_heads,
        n_layer,
        causal,
        dropout=0.0,
        use_bias=False,
        block_size=1e5,
        
    ):
        super().__init__()

        assert vocab_size is not None
        assert block_size is not None
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(vocab_size, n_embd),
            wpe = nn.Embedding(block_size, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([TransformerBlock(n_embd, n_heads, block_size, dropout, use_bias, causal) for _ in range(n_layer)]),
            ln_f = nn.LayerNorm(n_embd, bias=use_bias),
        ))
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        self.transformer.wte.weight = self.lm_head.weight # Weight Tying paper

        # init all weights
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
    
    def get_num_params(self, non_embedding=False):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)


    def forward(self, input):
        device = input.device
        b, t = input.size()
        assert t <= self.transformer.wpe.weight.size(0), f"Cannot forward sequence of length {t}, block size is only {self.transformer.wpe.weight.size(0)}"

        tok_emb = self.transformer.wte(input)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)

        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)

        return x

    def cross_entropy(self, input):
        x = self(input)
        return F.cross_entropy(x.transpose(1, 2), input)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # Any parameters that is 2D will be weight decayed, otherwise no.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer