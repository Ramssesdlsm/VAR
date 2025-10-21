"""
Script para generar imágenes usando el modelo VAR entrenado.
Basado en el repositorio original FoundationVision/VAR.

Uso:
    python generate_images.py --checkpoint ./output/var_d12/ar-ckpt-best.pth \
                               --num_images 8 \
                               --cfg 1.5 \
                               --seed 42 \
                               --output_dir ./generated_samples
"""

import argparse
import os
from pathlib import Path

import torch
import torchvision
from PIL import Image as PImage
import numpy as np

from models import VQVAE, VAR


def build_var_model(depth=12, embed_dim=768, num_heads=12, device='cuda'):
    """Construye el modelo VAR con la arquitectura especificada."""
    
    # Configuración del VQ-VAE (fixed hyperparameters)
    vqvae = VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        dropout=0.0,
        beta=0.25,
        using_znorm=False,
        quant_conv_ks=3,
        quant_resi=0.5,
        share_quant_resi=4,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        test_mode=True
    ).to(device)
    
    # Configuración del transformer
    # VARTransformer espera estos parámetros específicos
    transformer_config = {
        'depth': depth,
        'dim': embed_dim,  # 'dim' no 'embed_dim'
        'num_heads': num_heads,
        'mlp_ratio': 4.0,
        'attn_dropout': 0.0,
        'out_dropout': 0.0,
        'ffn_dropout': 0.0,
        'num_classes': 1000,
        'vocab_size': 4096,
        'max_seq_len': 680,  # Suma de patch_nums^2: 1+4+9+16+25+36+64+100+169+256=680
        'num_levels': 10,  # Número de escalas
        'first_scale_tokens': 1,  # patch_nums[0]^2 = 1^2 = 1
        'Cvae': 32,  # Dimensión del VQ-VAE
    }
    
    # Crear modelo VAR
    var_model = VAR(vqvae_model=vqvae, transformer_config=transformer_config).to(device)
    
    return var_model, vqvae


def load_checkpoint(model, vae, checkpoint_path, device='cuda'):
    """Carga los pesos del checkpoint."""
    print(f"[Loading] Cargando checkpoint desde {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # El checkpoint puede tener diferentes estructuras
    if 'var' in checkpoint:
        # Estructura del entrenamiento distribuido (DDP)
        model.load_state_dict(checkpoint['var'], strict=False)
    elif 'module' in checkpoint:
        # Estructura DDP directa
        state_dict = checkpoint['module']
        # Remover prefijo 'module.' si existe
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
    else:
        # Estructura simple
        model.load_state_dict(checkpoint, strict=False)
    
    print("[Loading] ✓ Checkpoint cargado exitosamente")
    return model


def generate_images(
    model,
    num_images=8,
    class_labels=None,
    cfg_scale=1.5,
    top_k=900,
    top_p=0.96,
    seed=None,
    more_smooth=False,
    device='cuda'
):
    """
    Genera imágenes usando el modelo VAR.
    
    Args:
        model: Modelo VAR
        num_images: Número de imágenes a generar
        class_labels: Lista de labels ImageNet (0-999). Si es None, se samplea aleatoriamente
        cfg_scale: Classifier-free guidance scale (1.5 recomendado para FID)
        top_k: Top-k sampling (900 recomendado)
        top_p: Top-p sampling (0.96 recomendado)
        seed: Semilla para reproducibilidad
        more_smooth: Usar Gumbel softmax (solo para visualización)
        device: Device CUDA
    
    Returns:
        Tensor de imágenes [N, 3, H, W] en rango [0, 1]
    """
    model.eval()
    
    # Preparar labels
    if class_labels is None:
        label_B = None  # Se samplearán aleatoriamente
    elif isinstance(class_labels, int):
        label_B = class_labels
    else:
        label_B = torch.tensor(class_labels, device=device, dtype=torch.long)
        num_images = len(class_labels)
    
    print(f"\n[Generation] Generando {num_images} imágenes...")
    print(f"[Generation] CFG scale: {cfg_scale}")
    print(f"[Generation] Top-k: {top_k}, Top-p: {top_p}")
    if seed is not None:
        print(f"[Generation] Seed: {seed}")
    
    # Generar imágenes
    with torch.no_grad():
        generated_images = model.autoregressive_infer_cfg(
            B=num_images,
            label_B=label_B,
            g_seed=seed,
            cfg=cfg_scale,
            top_k=top_k,
            top_p=top_p,
            more_smooth=more_smooth
        )
    
    print(f"[Generation] ✓ Generación completada")
    print(f"[Generation] Shape: {generated_images.shape}, Range: [{generated_images.min():.3f}, {generated_images.max():.3f}]")
    
    return generated_images


def save_images(images, output_dir, prefix='sample', labels=None):
    """
    Guarda las imágenes generadas.
    
    Args:
        images: Tensor [N, 3, H, W] en rango [0, 1]
        output_dir: Directorio de salida
        prefix: Prefijo para los nombres de archivo
        labels: Labels de clase (opcional, para incluir en el nombre)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Guardar grid completo
    grid_path = os.path.join(output_dir, f'{prefix}_grid.png')
    grid = torchvision.utils.make_grid(images, nrow=min(8, len(images)), padding=2, pad_value=1.0)
    grid_np = grid.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
    PImage.fromarray(grid_np).save(grid_path)
    print(f"[Save] Grid guardado en: {grid_path}")
    
    # Guardar imágenes individuales
    for i, img in enumerate(images):
        if labels is not None and i < len(labels):
            filename = f'{prefix}_class{labels[i]:03d}_{i:04d}.png'
        else:
            filename = f'{prefix}_{i:04d}.png'
        
        img_path = os.path.join(output_dir, filename)
        img_np = img.permute(1, 2, 0).mul_(255).cpu().numpy().astype(np.uint8)
        PImage.fromarray(img_np).save(img_path)
    
    print(f"[Save] ✓ {len(images)} imágenes guardadas en: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generar imágenes con VAR')
    
    # Paths
    parser.add_argument('--vae_ckpt', type=str, default='vae_ch160v4096z32.pth',
                       help='Path al checkpoint del VQ-VAE')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path al checkpoint del modelo VAR entrenado')
    parser.add_argument('--output_dir', type=str, default='./generated_samples',
                       help='Directorio para guardar las imágenes generadas')
    
    # Arquitectura
    parser.add_argument('--depth', type=int, default=12,
                       help='Profundidad del transformer (3, 12, 16, 20, 24, 30)')
    parser.add_argument('--embed_dim', type=int, default=768,
                       help='Dimensión de embeddings (768 para d=12)')
    parser.add_argument('--num_heads', type=int, default=12,
                       help='Número de attention heads (12 para d=12)')
    
    # Generación
    parser.add_argument('--num_images', type=int, default=8,
                       help='Número de imágenes a generar')
    parser.add_argument('--class_labels', type=int, nargs='+', default=None,
                       help='Labels de clase ImageNet (0-999). Ej: 22 437 980')
    parser.add_argument('--cfg', type=float, default=1.5,
                       help='Classifier-free guidance scale (1.5 para FID, 4-5 para calidad visual)')
    parser.add_argument('--top_k', type=int, default=900,
                       help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.96,
                       help='Top-p (nucleus) sampling')
    parser.add_argument('--seed', type=int, default=None,
                       help='Semilla para reproducibilidad')
    parser.add_argument('--more_smooth', action='store_true',
                       help='Usar Gumbel softmax para imágenes más suaves (solo visualización)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda o cpu)')
    
    args = parser.parse_args()
    
    # Verificar CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("[Warning] CUDA no disponible, usando CPU")
        args.device = 'cpu'
    
    print("="*80)
    print("  VAR Image Generation")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Modelo: depth={args.depth}, embed_dim={args.embed_dim}, num_heads={args.num_heads}")
    print(f"Checkpoint: {args.checkpoint}")
    print("="*80)
    
    # Construir modelo
    print("\n[Build] Construyendo modelo VAR...")
    model, vae = build_var_model(
        depth=args.depth,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        device=args.device
    )
    print(f"[Build] ✓ Modelo construido")
    
    # Cargar VQ-VAE
    if os.path.exists(args.vae_ckpt):
        print(f"\n[VAE] Cargando VQ-VAE desde {args.vae_ckpt}")
        vae.load_state_dict(torch.load(args.vae_ckpt, map_location=args.device), strict=True)
        print("[VAE] ✓ VQ-VAE cargado")
    else:
        print(f"[Warning] VQ-VAE checkpoint no encontrado: {args.vae_ckpt}")
        print("[Warning] Usando pesos inicializados (no recomendado)")
    
    # Cargar checkpoint del VAR
    model = load_checkpoint(model, vae, args.checkpoint, args.device)
    
    # Generar imágenes
    generated_images = generate_images(
        model=model,
        num_images=args.num_images,
        class_labels=args.class_labels,
        cfg_scale=args.cfg,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        more_smooth=args.more_smooth,
        device=args.device
    )
    
    # Guardar imágenes
    save_images(
        images=generated_images,
        output_dir=args.output_dir,
        prefix='var_sample',
        labels=args.class_labels
    )
    
    print("\n" + "="*80)
    print("✓ Generación completada exitosamente!")
    print("="*80)


if __name__ == '__main__':
    main()
