from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vqvae import VQVAE
from models.transformers import VARTransformer

def prepare_var_inputs(token_maps, labels, patch_nums):
    b = labels.shape[0]
    flat_tokens = [t.view(b, -1) for t in token_maps]
    full_seq = torch.cat(flat_tokens, dim=1)
    
    first_scale_tokens = patch_nums[0] ** 2
    teacher_forcing_tokens = full_seq[:, first_scale_tokens:]
    
    level_indices_list = [torch.full((pn**2,), i, dtype=torch.long) for i, pn in enumerate(patch_nums)]
    level_indices = torch.cat(level_indices_list).to(labels.device)
    
    d = level_indices
    attn_mask = d.unsqueeze(1) >= d.unsqueeze(0)
    
    return {
        "teacher_forcing_tokens": teacher_forcing_tokens,
        "class_labels": labels,
        "level_indices": level_indices.unsqueeze(0).expand(b, -1),
        "attn_mask": attn_mask,
    }

class VAR(nn.Module):
    """
    Clase unificada que encapsula el VQ-VAE (Etapa 1) y el VARTransformer (Etapa 2).
    """
    def __init__(self, vqvae_model: VQVAE, transformer_config: Dict = None):
        super().__init__()
        # --- Etapa 1: VQ-VAE ---
        self.vqvae = vqvae_model
        # Congelamos el VQ-VAE, no se entrenará
        for param in self.vqvae.parameters():
            param.requires_grad = False
        
        self.patch_nums = self.vqvae.patch_nums

        # --- Etapa 2: VARTransformer ---
        if transformer_config is None:
            transformer_config = {}
        
        self.transformer = VARTransformer(**transformer_config)
        print("Modelo VAR unificado creado exitosamente.")

    def forward(self, images: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """
        Paso hacia adelante para el ENTRENAMIENTO (usa Teacher Forcing).

        Args:
            images (torch.Tensor): Lote de imágenes de entrada (B, 3, H, W).
            labels (torch.LongTensor): Lote de etiquetas de clase (B,).

        Returns:
            torch.Tensor: Logits predichos por el transformador.
        """
        # 1. Tokenizar las imágenes usando el VQ-VAE (sin gradientes y sin autocast)
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            # Forzar float32 para el VAE congelado
            images_fp32 = images.float()
            token_maps = self.vqvae.encode(images_fp32)
        
        # 2. Preparar los tensores de entrada para el transformador
        model_inputs = prepare_var_inputs(token_maps, labels, self.patch_nums)
        
        # 3. Pasar los tensores al transformador para obtener los logits
        logits = self.transformer(**model_inputs)
        
        return logits

    # Innoperativo
    @torch.no_grad()
    def sample(self, labels: torch.LongTensor, cfg_scale: float = 4.0, 
             top_k: int = 2048, device: str = 'cuda'):
        """
        Genera imágenes a partir de etiquetas de clase (INFERENCIA).
        """
        print(f"Iniciando muestreo autorregresivo para {len(labels)} imágenes.")
        self.eval()
        n = len(labels)
        
        # Preparar para CFG: crear etiquetas nulas
        null_labels = torch.full_like(labels, self.transformer.class_embedding.num_embeddings - 1)
        cfg_labels = torch.cat([labels, null_labels])
        
        # El `sos` es el embedding de clase
        class_cond = self.transformer.class_embedding(cfg_labels)

        # Generación escala por escala
        generated_tokens_per_scale = []
        current_sequence = []
        
        for i, pn in enumerate(self.patch_nums):
            num_tokens_in_scale = pn ** 2
            
            # Preparar la entrada para la escala actual
            if i == 0:
                # La primera escala solo usa el prompt `sos`
                input_seq = class_cond.unsqueeze(1).expand(-1, num_tokens_in_scale, -1) + \
                            self.transformer.pos_start + \
                            self.transformer.level_embedding(torch.tensor([i], device=device))
            else:
                # Las escalas posteriores usan los tokens generados previamente
                flat_prev_tokens = torch.cat(current_sequence, dim=1)
                
                # --- Lógica de `sos` + `teacher_forcing` para inferencia ---
                teacher_tokens_emb = self.transformer.token_embedding(flat_prev_tokens)
                
                sos_tokens = class_cond.unsqueeze(1).expand(-1, self.patch_nums[0]**2, -1)
                sos_lvl_indices = torch.zeros(self.patch_nums[0]**2, dtype=torch.long, device=device)
                sos_seq = sos_tokens + self.transformer.pos_start + self.transformer.level_embedding(sos_lvl_indices)

                total_len = sos_seq.shape[1] + teacher_tokens_emb.shape[1]
                
                pos_emb = self.transformer.position_embedding[:, self.patch_nums[0]**2:total_len, :]
                
                lvl_indices_list = []
                for j, prev_pn in enumerate(self.patch_nums[:i]):
                   lvl_indices_list.append(torch.full((prev_pn**2,), j, dtype=torch.long, device=device))
                lvl_indices = torch.cat(lvl_indices_list)

                teacher_seq = teacher_tokens_emb + pos_emb + self.transformer.level_embedding(lvl_indices)
                
                input_seq = torch.cat([sos_seq, teacher_seq], dim=1)

            # Crear la máscara de atención
            d = torch.cat([torch.full((p**2,), j, device=device) for j, p in enumerate(self.patch_nums[:i+1])])
            mask = d.unsqueeze(1) >= d.unsqueeze(0)
            
            # Obtener logits del modelo
            logits = self.transformer.head(
                self.transformer.blocks[-1]( # Simplificación para la inferencia
                    input_seq, 
                    class_cond,
                    mask),
                class_cond
            )[:, -num_tokens_in_scale:] # Solo nos importan los logits de la escala actual

            # Aplicar CFG
            logits_cond, logits_uncond = logits.chunk(2)
            logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)

            # Muestreo Top-K
            top_k_values, top_k_indices = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
            probs = F.softmax(top_k_values, dim=-1)
            sampled_indices = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(n, num_tokens_in_scale)
            sampled_tokens = torch.gather(top_k_indices, -1, sampled_indices.unsqueeze(-1)).squeeze(-1)
            
            generated_tokens_per_scale.append(sampled_tokens.view(n, pn, pn))
            current_sequence.append(sampled_tokens)
        
        # Decodificar los tokens generados a imágenes
        print("Decodificando tokens a imágenes...")
        generated_images = self.vqvae.decode(generated_tokens_per_scale)
        self.train()
        return generated_images