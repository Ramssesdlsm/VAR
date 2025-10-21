from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vqvae import VQVAE
from models.transformers import VARTransformer

def prepare_var_inputs(token_maps, labels, patch_nums, vqvae_quantizer):
    b = labels.shape[0]
    
    # Get processed embeddings from VQ-VAE quantizer
    x_BLCv_wo_first_l = vqvae_quantizer.idxBl_to_var_input(token_maps)
    
    level_indices_list = [torch.full((pn**2,), i, dtype=torch.long) for i, pn in enumerate(patch_nums)]
    level_indices = torch.cat(level_indices_list).to(labels.device)
    
    d = level_indices
    
    attn_mask = d.unsqueeze(1) >= d.unsqueeze(0)
    
    return {
        "x_BLCv_wo_first_l": x_BLCv_wo_first_l,
        "class_labels": labels,
        "level_indices": level_indices.unsqueeze(0).expand(b, -1),
        "attn_mask": attn_mask,
    }

class VAR(nn.Module):
    def __init__(self, vqvae_model: VQVAE, transformer_config: Dict = None):
        super().__init__()

        self.vqvae = vqvae_model
        # Freeze VQ-VAE parameters
        for param in self.vqvae.parameters():
            param.requires_grad = False
        
        self.patch_nums = self.vqvae.patch_nums

        if transformer_config is None:
            transformer_config = {}
        
        self.transformer = VARTransformer(**transformer_config)

    def forward(self, images: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            images_fp32 = images.float()
            token_maps = self.vqvae.img_to_idxBl(images_fp32)

        model_inputs = prepare_var_inputs(token_maps, labels, self.patch_nums, self.vqvae.quantize)
        
        logits = self.transformer(**model_inputs)
        
        return logits

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        B: int,
        label_B: Optional[torch.LongTensor] = None,
        g_seed: Optional[int] = None,
        cfg: float = 1.5,
        top_k: int = 900,
        top_p: float = 0.96,
        more_smooth: bool = False
    ) -> torch.Tensor:

        from models.helpers import sample_with_top_k_top_p_, gumbel_softmax_with_rng
        
        self.eval()
        device = next(self.parameters()).device
        
        # Setup random generator
        if g_seed is not None:
            rng = torch.Generator(device=device).manual_seed(g_seed)
        else:
            rng = None
        
        # Handle labels
        num_classes = self.transformer.class_embedding.num_embeddings - 1  # -1 for unconditional
        if label_B is None:
            # Sample random classes
            label_B = torch.randint(0, num_classes, (B,), device=device, generator=rng)
        elif isinstance(label_B, int):
            # Single class for all samples
            label_B = torch.full((B,), label_B, device=device)
        else:
            # Ensure on correct device
            label_B = label_B.to(device)
        
        # Create conditional and unconditional labels for CFG
        # Unconditional uses num_classes (last embedding index)
        unc_label_B = torch.full_like(label_B, num_classes)
        cfg_label_B = torch.cat([label_B, unc_label_B])  # [2B]
        
        # Get class embeddings for CFG
        sos = cond_BD = self.transformer.class_embedding(cfg_label_B)  # [2B, D]
        
        # Prepare positional and level embeddings
        L = sum(pn ** 2 for pn in self.patch_nums)
        lvl_pos = self.transformer.level_embedding(
            torch.cat([torch.full((pn**2,), i, dtype=torch.long, device=device) 
                      for i, pn in enumerate(self.patch_nums)])
        ) + self.transformer.position_embedding.expand(2*B, -1, -1)  # [2B, L, D]
        
        # Initialize next_token_map for first scale
        first_l = self.patch_nums[0] ** 2
        next_token_map = (
            sos.unsqueeze(1).expand(2*B, first_l, -1) + 
            lvl_pos[:, :first_l]
        )  # [2B, first_l, D]
        
        # Initialize f_hat accumulator
        Cvae = self.vqvae.Cvae
        f_hat = torch.zeros(B, Cvae, self.patch_nums[-1], self.patch_nums[-1], device=device)
        
        cur_L = 0
        
        # Autoregressive generation over scales
        for si, pn in enumerate(self.patch_nums):
            ratio = si / (len(self.patch_nums) - 1)
            cur_L += pn * pn
            
            # Forward through transformer
            x = next_token_map
            for block in self.transformer.blocks:
                x = block(x, cond=cond_BD, attn_mask=None)
            
            # Get logits for current scale
            logits_BlV = self.transformer.head(x, cond_BD)  # [2B, pn*pn, vocab_size]
            
            # Apply CFG
            t = cfg * ratio
            logits_cond, logits_uncond = logits_BlV.chunk(2)
            logits_BlV = (1 + t) * logits_cond - t * logits_uncond  # [B, pn*pn, vocab_size]
            
            # Sample tokens
            if not more_smooth:
                # Standard sampling (used for FID evaluation)
                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
                )[:, :, 0]  # [B, pn*pn]
                h_BChw = self.vqvae.quantize.embedding(idx_Bl)  # [B, pn*pn, Cvae]
            else:
                # Gumbel softmax for smoother visualization
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                ) @ self.vqvae.quantize.embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose(1, 2).reshape(B, Cvae, pn, pn)  # [B, Cvae, pn, pn]
            
            # Update f_hat and prepare next input
            f_hat, next_token_map_pre = self.vqvae.quantize.get_next_autoregressive_input(
                si, len(self.patch_nums), f_hat, h_BChw
            )
            
            if si != len(self.patch_nums) - 1:
                # Prepare input for next scale
                next_token_map_pre = next_token_map_pre.view(B, Cvae, -1).transpose(1, 2)  # [B, next_l, Cvae]
                next_token_map = self.transformer.word_embed(next_token_map_pre) + \
                                lvl_pos[:B, cur_L:cur_L + self.patch_nums[si+1]**2]
                next_token_map = next_token_map.repeat(2, 1, 1)  # [2B, next_l, D] for CFG
        
        # Decode f_hat to image
        generated_images = self.vqvae.fhat_to_img(f_hat)  # [B, 3, H, W] in [-1, 1]
        generated_images = generated_images.add_(1).mul_(0.5)  # Convert to [0, 1]
        
        return generated_images