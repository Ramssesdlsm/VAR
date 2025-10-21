"""
Script simple para generar imágenes de ejemplo.
Usa el checkpoint entrenado y genera algunas imágenes de clases específicas de ImageNet.

Clases de ejemplo comunes:
- 22: European fire salamander (salamandra)
- 207: Golden retriever (perro)
- 281: Tabby cat (gato)
- 283: Persian cat (gato persa)
- 437: Baseball (béisbol)
- 562: Fountain (fuente)
- 980: Volcano (volcán)
- 970: Alp (montaña)
"""

import torch
from models import VQVAE, VAR
from PIL import Image
import torchvision
import numpy as np


def main():
    print("="*80)
    print("  VAR - Generación Simple de Imágenes")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # ==================== 1. Construir modelo ====================
    print("[1/4] Construyendo modelo VAR...")
    
    # VQ-VAE
    vqvae = VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        test_mode=True
    ).to(device)
    
    # Transformer config - parámetros correctos para VARTransformer
    transformer_config = {
        'depth': 12,         # d=12
        'dim': 768,          # 768 para d=12
        'num_heads': 12,     # 12 para d=12
        'mlp_ratio': 4.0,
        'attn_dropout': 0.0,
        'out_dropout': 0.0,
        'ffn_dropout': 0.0,
        'num_classes': 1000,
        'vocab_size': 4096,
        'max_seq_len': 680,  # Suma de patch_nums^2
        'num_levels': 10,
        'first_scale_tokens': 1,
        'Cvae': 32,
    }
    
    # Modelo VAR
    model = VAR(vqvae_model=vqvae, transformer_config=transformer_config).to(device)
    print("✓ Modelo construido")
    
    # ==================== 2. Cargar checkpoints ====================
    print("\n[2/4] Cargando checkpoints...")
    
    # Cargar VQ-VAE
    vae_path = 'vae_ch160v4096z32.pth'
    try:
        vqvae.load_state_dict(torch.load(vae_path, map_location=device), strict=True)
        print(f"✓ VQ-VAE cargado desde {vae_path}")
    except FileNotFoundError:
        print(f"⚠ Warning: {vae_path} no encontrado")
        print("  Descarga desde: https://huggingface.co/FoundationVision/var/resolve/main/vae_ch160v4096z32.pth")
    
    # Cargar VAR entrenado
    var_ckpt_path = './output/var_d12/checkpoint-0079.pth'  # Ajusta esta ruta
    try:
        ckpt = torch.load(var_ckpt_path, map_location=device)
        
        # Intentar diferentes estructuras de checkpoint
        if 'var' in ckpt:
            model.load_state_dict(ckpt['var'], strict=False)
        elif 'module' in ckpt:
            state_dict = {k.replace('module.', ''): v for k, v in ckpt['module'].items()}
            model.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        
        print(f"✓ VAR cargado desde {var_ckpt_path}")
        
        # Mostrar info del checkpoint
        if 'epoch' in ckpt:
            print(f"  Época: {ckpt['epoch']}")
        if 'args' in ckpt and 'vacc_mean' in ckpt['args']:
            print(f"  Accuracy: {ckpt['args']['vacc_mean']:.2f}%")
            
    except FileNotFoundError:
        print(f"⚠ Error: Checkpoint no encontrado en {var_ckpt_path}")
        print("  Ajusta la ruta en el script o especifica con --checkpoint")
        return
    
    model.eval()
    
    # ==================== 3. Generar imágenes ====================
    print("\n[3/4] Generando imágenes...")
    
    # Configuración de generación
    class_labels = [207, 281, 437, 562, 980, 970, 22, 283]  # 8 clases diferentes
    num_images = len(class_labels)
    cfg_scale = 4.0      # 4.0 para mejor calidad visual, 1.5 para FID
    seed = 42
    
    print(f"  Clases: {class_labels}")
    print(f"  CFG scale: {cfg_scale}")
    print(f"  Seed: {seed}")
    
    with torch.no_grad():
        generated = model.autoregressive_infer_cfg(
            B=num_images,
            label_B=torch.tensor(class_labels, device=device),
            g_seed=seed,
            cfg=cfg_scale,
            top_k=900,
            top_p=0.96,
            more_smooth=False
        )
    
    print(f"✓ Generadas {num_images} imágenes")
    print(f"  Shape: {generated.shape}")
    print(f"  Range: [{generated.min():.3f}, {generated.max():.3f}]")
    
    # ==================== 4. Guardar resultados ====================
    print("\n[4/4] Guardando imágenes...")
    
    # Crear grid
    grid = torchvision.utils.make_grid(generated, nrow=4, padding=4, pad_value=1.0)
    grid_np = grid.permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
    
    # Guardar grid
    grid_img = Image.fromarray(grid_np)
    grid_path = 'generated_grid.png'
    grid_img.save(grid_path)
    print(f"✓ Grid guardado: {grid_path}")
    
    # Guardar individuales
    class_names = {
        22: 'salamander', 207: 'golden_retriever', 281: 'tabby_cat',
        283: 'persian_cat', 437: 'baseball', 562: 'fountain',
        980: 'volcano', 970: 'alp'
    }
    
    for i, (img, label) in enumerate(zip(generated, class_labels)):
        img_np = img.permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8)
        name = class_names.get(label, f'class{label}')
        img_path = f'generated_{i}_{name}.png'
        Image.fromarray(img_np).save(img_path)
    
    print(f"✓ {num_images} imágenes individuales guardadas")
    
    print("\n" + "="*80)
    print("✓ ¡Generación completada!")
    print("="*80)
    print(f"\nRevisa las imágenes generadas:")
    print(f"  - Grid completo: {grid_path}")
    print(f"  - Imágenes individuales: generated_*.png")


if __name__ == '__main__':
    main()
