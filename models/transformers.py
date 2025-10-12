import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.0):
       super(FeedForwardNetwork, self).__init__()
       self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(dropout)
       )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       return self.net(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.0, out_dropout: float = 0.0):
        super(MultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, "The dimension must be divisible by the number of heads"

        self.dim = dim
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

class AdaptiveLayerNorm(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super(AdaptiveLayerNorm, self).__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * dim, bias=True)
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_norm = self.norm(x)

        params = self.modulation(cond)

        params = params.unsqueeze(1)

        gate_attn, gate_ffn, shift_attn, scale_attn, shift_ffn, scale_ffn = params.chunk(6, dim=-1)

        x_modulated_attn = x_norm * (1 + scale_attn) + shift_attn
        x_modulated_ffn = x_norm * (1 + scale_ffn) + shift_ffn

        return x_modulated_attn, gate_attn, x_modulated_ffn, gate_ffn
    
class AdaptiveLayerNormBeforeHead(nn.Module):
    def __init__(self, dim: int, cond_dim: int, vocab_size: int):
        super(AdaptiveLayerNormBeforeHead, self).__init__()

        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * dim, bias=True)
        )

        self.output = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(cond).chunk(2, dim=-1)

        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        x = self.norm(x) * (1 + scale) + shift

        logits = self.output(x)

        return logits

class VARBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0, attn_dropout: float = 0.0, out_dropout: float = 0.0, ffn_dropout: float = 0.0):
        super(VARBlock, self).__init__()

        self.norm1 = AdaptiveLayerNorm(dim, cond_dim)

        self.attention = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
            out_dropout=out_dropout
        )

        self.norm2 = AdaptiveLayerNorm(dim, cond_dim)

        self.ffn = FeedForwardNetwork(
            input_dim=dim,
            hidden_dim=round(dim * mlp_ratio),
            output_dim=dim,
            dropout=ffn_dropout
        )
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x_modulated_attn, gate_attn, _, _ = self.norm1(x, cond)

        attn_output = self.attention(x_modulated_attn, attn_mask)

        x = x + gate_attn * attn_output

        _, _, x_modulated_ffn, gate_ffn = self.norm2(x, cond)

        ffn_output = self.ffn(x_modulated_ffn)

        x = x + gate_ffn * ffn_output

        return x
    
class VARTransformer(nn.Module):
    def __init__(self, depth: int, dim: int, num_heads: int, mlp_ratio: float, attn_dropout: float, out_dropout: float, ffn_dropout: float, num_classes: int, vocab_size: int, max_seq_len: int, num_levels: int, first_scale_tokens: int):
        super(VARTransformer, self).__init__()

        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.first_scale_tokens = first_scale_tokens

        # Embeddings
        self.class_embedding = nn.Embedding(num_classes + 1, dim)
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_seq_len, dim))
        self.level_embedding = nn.Embedding(num_levels, dim)
        self.pos_start = nn.Parameter(torch.zeros(1, first_scale_tokens, dim))

        # Backbone
        self.blocks = nn.ModuleList([
            VARBlock(
                dim=dim,
                num_heads=num_heads,
                cond_dim=dim,
                mlp_ratio=mlp_ratio,
                attn_dropout=attn_dropout,
                out_dropout=out_dropout,
                ffn_dropout=ffn_dropout
            ) for _ in range(depth)
        ])

        # Head
        self.head = AdaptiveLayerNormBeforeHead(dim, dim, vocab_size)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.pos_start, std=0.02)
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        self.apply(_init_weights)
    
    def forward(self, teacher_forcing_tokens: torch.Tensor, class_labels: torch.Tensor, level_indices: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass para VAR con teacher forcing.
        
        Args:
            teacher_forcing_tokens: tokens de scales 1-N (sin scale 0), shape [B, L-first_scale_tokens]
            class_labels: etiquetas de clase, shape [B]
            level_indices: índices de nivel para cada token, shape [B, L]
            attn_mask: máscara de atención causal, shape [L, L]
        
        Returns:
            logits: predicciones para TODAS las escalas 0-N, shape [B, L, vocab_size]
        """
        seq_len = teacher_forcing_tokens.shape[1] + self.first_scale_tokens
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds model capacity {self.max_seq_len}"

        class_cond = self.class_embedding(class_labels)

        # Scale 0: usa class embedding como "token" 
        sos_tokens = class_cond.unsqueeze(1).expand(-1, self.first_scale_tokens, -1)
        sos_level_indices = torch.zeros(self.first_scale_tokens, dtype=torch.long, device=class_cond.device)
        # Usar position_embedding consistente para todas las posiciones
        sos_sequence = sos_tokens + self.position_embedding[:, :self.first_scale_tokens, :] + self.level_embedding(sos_level_indices)

        # Scales 1-N: usa embeddings de los tokens reales
        token_embeddings = self.token_embedding(teacher_forcing_tokens)
        position_embeddings = self.position_embedding[:, self.first_scale_tokens:seq_len, :]
        level_embeddings = self.level_embedding(level_indices[:, self.first_scale_tokens:])
        teacher_sequence = token_embeddings + position_embeddings + level_embeddings

        # Concatenar: [scale0_representations, scale1-N_representations]
        x = torch.cat((sos_sequence, teacher_sequence), dim=1)

        # Aplicar transformer blocks con máscara causal
        for block in self.blocks:
            x = block(x, cond=class_cond, attn_mask=attn_mask)
        
        # Generar logits para todas las posiciones
        logits = self.head(x, cond=class_cond)

        return logits