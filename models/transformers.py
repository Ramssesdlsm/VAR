import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, out_dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "The dimension must be divisible by the number of heads"

        self.num_head = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)

        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.register_buffer('k_bias', torch.zeros(dim))

        self.to_out = nn.Linear(dim, dim, bias=True)
        
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = F.linear(input=x, weight=self.to_qkv.weight, bias=torch.cat((self.q_bias, self.k_bias, self.v_bias)))

        # (B, L, 3*dim) -> (B, L, 3, H, HD)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_head, self.head_dim)

        # (B, L, 3, H, HD) -> (3, B, H, L, HD)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=False
        )

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)

        out = self.out_dropout(self.to_out(out))

        return out
