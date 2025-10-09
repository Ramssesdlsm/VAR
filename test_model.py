#!/usr/bin/env python3
"""
Script de ejemplo simple para probar el entrenamiento de VAR.
Úsalo para depuración rápida antes de lanzar entrenamientos largos.
"""

import os
import torch
from models import VAR, VQVAE

def test_forward_pass():
    """Prueba un forward pass del modelo VAR."""
    print("="*60)
    print("Probando forward pass del modelo VAR")
    print("="*60)
    
    # Configuración
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 2
    
    # 1. Cargar VQ-VAE
    print("\n[1/4] Cargando VQ-VAE...")
    vae_ckpt = 'vae_ch160v4096z32.pth'
    if not os.path.exists(vae_ckpt):
        print(f"ERROR: No se encontró {vae_ckpt}")
        print("Descarga el checkpoint del VQ-VAE primero.")
        return False
    
    vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True)
    checkpoint = torch.load(vae_ckpt, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    vae.load_state_dict(state_dict, strict=True)
    vae = vae.to(device).eval()
    print(f"✓ VQ-VAE cargado - vocab_size={vae.vocab_size}")
    
    # 2. Crear modelo VAR
    print("\n[2/4] Creando modelo VAR...")
    # Calcular max_seq_len basado en patch_nums
    max_seq_len = sum(pn * pn for pn in vae.patch_nums)
    
    transformer_config = {
        'depth': 16,
        'dim': 1024,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'attn_dropout': 0.0,
        'out_dropout': 0.0,
        'ffn_dropout': 0.0,
        'num_classes': 1000,
        'vocab_size': vae.vocab_size,
        'max_seq_len': max_seq_len,
        'num_levels': len(vae.patch_nums),
        'first_scale_tokens': vae.patch_nums[0] ** 2,
    }
    var_model = VAR(vqvae_model=vae, transformer_config=transformer_config)
    var_model = var_model.to(device)
    
    total_params = sum(p.numel() for p in var_model.parameters())
    trainable_params = sum(p.numel() for p in var_model.parameters() if p.requires_grad)
    print(f"✓ VAR creado - total_params={total_params/1e6:.2f}M, trainable={trainable_params/1e6:.2f}M")
    
    # 3. Crear batch de prueba
    print("\n[3/4] Creando batch de prueba...")
    images = torch.randn(batch_size, 3, 256, 256).to(device)
    labels = torch.randint(0, 1000, (batch_size,)).to(device)
    print(f"✓ Batch creado - images: {images.shape}, labels: {labels.shape}")
    
    # 4. Forward pass
    print("\n[4/4] Ejecutando forward pass...")
    var_model.train()
    try:
        with torch.cuda.amp.autocast(enabled=False):
            logits = var_model(images, labels)
        print(f"✓ Forward exitoso - logits: {logits.shape}")
        
        # Calcular loss de ejemplo
        with torch.no_grad():
            token_maps = vae.encode(images)
            gt_tokens = torch.cat([t.view(batch_size, -1) for t in token_maps], dim=1)
        
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vae.vocab_size),
            gt_tokens.view(-1)
        )
        print(f"✓ Loss calculado: {loss.item():.4f}")
        
        print("\n" + "="*60)
        print("✓ ¡PRUEBA EXITOSA!")
        print("="*60)
        print("\nEl modelo está listo para entrenamiento.")
        print("Ejecuta: python train.py --data_path /path/to/imagenet ...")
        return True
        
    except Exception as e:
        print(f"\n✗ Error durante forward pass:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sampling():
    """Prueba el muestreo autorregresivo (generación de imágenes)."""
    print("\n" + "="*60)
    print("Probando muestreo autorregresivo")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Cargar modelos
    print("\n[1/3] Cargando modelos...")
    vae_ckpt = 'vae_ch160v4096z32.pth'
    if not os.path.exists(vae_ckpt):
        print(f"ERROR: No se encontró {vae_ckpt}")
        return False
    
    vae = VQVAE(vocab_size=4096, z_channels=32, ch=160, test_mode=True)
    checkpoint = torch.load(vae_ckpt, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    vae.load_state_dict(state_dict, strict=True)
    vae = vae.to(device).eval()
    
    # Calcular max_seq_len basado en patch_nums
    max_seq_len = sum(pn * pn for pn in vae.patch_nums)
    
    transformer_config = {
        'depth': 16,
        'dim': 1024,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'attn_dropout': 0.0,
        'out_dropout': 0.0,
        'ffn_dropout': 0.0,
        'num_classes': 1000,
        'vocab_size': vae.vocab_size,
        'max_seq_len': max_seq_len,
        'num_levels': len(vae.patch_nums),
        'first_scale_tokens': vae.patch_nums[0] ** 2,
    }
    var_model = VAR(vqvae_model=vae, transformer_config=transformer_config)
    var_model = var_model.to(device).eval()
    print("✓ Modelos cargados")
    
    # Muestrear
    print("\n[2/3] Generando imágenes...")
    print("NOTA: Este modelo NO está entrenado, las imágenes serán aleatorias")
    labels = torch.tensor([207, 360, 387, 974]).to(device)  # Clases de ImageNet
    
    try:
        with torch.no_grad():
            images = var_model.sample(labels, cfg_scale=1.0, top_k=2048, device=device)
        print(f"✓ Imágenes generadas: {images.shape}")
        
        # Guardar
        print("\n[3/3] Guardando imágenes...")
        from torchvision.utils import save_image
        os.makedirs('samples', exist_ok=True)
        save_image(images, 'samples/test_generation.png', 
                  normalize=True, value_range=(-1, 1), nrow=2)
        print("✓ Imágenes guardadas en: samples/test_generation.png")
        
        print("\n" + "="*60)
        print("✓ ¡PRUEBA DE MUESTREO EXITOSA!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error durante muestreo:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Test VAR model')
    parser.add_argument('--test', default='forward', choices=['forward', 'sample', 'both'],
                       help='Tipo de prueba a ejecutar')
    args = parser.parse_args()
    
    success = True
    if args.test in ['forward', 'both']:
        success = test_forward_pass() and success
    
    if args.test in ['sample', 'both']:
        success = test_sampling() and success
    
    if success:
        print("\n✓ Todas las pruebas pasaron exitosamente")
        exit(0)
    else:
        print("\n✗ Algunas pruebas fallaron")
        exit(1)
