#!/usr/bin/env python3
"""
Script para verificar que el dataset ImageNet Localization se descargó correctamente.
"""

import os
import sys
from pathlib import Path

def check_dataset(data_path):
    """Verifica la estructura y contenido del dataset."""
    print("=" * 60)
    print("Verificación del Dataset ImageNet Localization")
    print("=" * 60)
    print()
    
    data_path = Path(data_path)
    
    if not data_path.exists():
        print(f"❌ ERROR: El directorio {data_path} no existe")
        print(f"   Ejecuta: ./scripts/download_imagenet_localization.sh")
        return False
    
    # Verificar directorios train y val
    train_dir = data_path / 'train'
    val_dir = data_path / 'val'
    
    checks_passed = 0
    total_checks = 5
    
    # Check 1: Directorio train existe
    print("[1/5] Verificando directorio train...")
    if train_dir.exists():
        print(f"      ✓ Existe: {train_dir}")
        checks_passed += 1
    else:
        print(f"      ❌ No existe: {train_dir}")
    
    # Check 2: Directorio val existe
    print("[2/5] Verificando directorio val...")
    if val_dir.exists():
        print(f"      ✓ Existe: {val_dir}")
        checks_passed += 1
    else:
        print(f"      ❌ No existe: {val_dir}")
    
    if not (train_dir.exists() and val_dir.exists()):
        print()
        print("❌ Directorios train/val no encontrados")
        return False
    
    # Check 3: Número de clases
    print("[3/5] Verificando número de clases...")
    train_classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    val_classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    
    print(f"      Train: {len(train_classes)} clases")
    print(f"      Val: {len(val_classes)} clases")
    
    if len(train_classes) == 1000 and len(val_classes) == 1000:
        print(f"      ✓ Ambos tienen 1000 clases")
        checks_passed += 1
    else:
        print(f"      ❌ Esperado 1000 clases, encontrado train={len(train_classes)}, val={len(val_classes)}")
    
    # Check 4: Contar imágenes en train
    print("[4/5] Contando imágenes en train...")
    train_images = list(train_dir.rglob('*.JPEG'))
    print(f"      Train: {len(train_images)} imágenes")
    
    if 35000 <= len(train_images) <= 45000:
        print(f"      ✓ Cantidad razonable (~40,000 esperadas)")
        checks_passed += 1
    else:
        print(f"      ⚠️  Cantidad inesperada (esperado ~40,000)")
    
    # Check 5: Contar imágenes en val
    print("[5/5] Contando imágenes en val...")
    val_images = list(val_dir.rglob('*.JPEG'))
    print(f"      Val: {len(val_images)} imágenes")
    
    if 8000 <= len(val_images) <= 12000:
        print(f"      ✓ Cantidad razonable (~10,000 esperadas)")
        checks_passed += 1
    else:
        print(f"      ⚠️  Cantidad inesperada (esperado ~10,000)")
    
    print()
    print("=" * 60)
    print(f"Resultado: {checks_passed}/{total_checks} verificaciones pasadas")
    print("=" * 60)
    
    # Verificación adicional con PyTorch
    print()
    print("Verificación con PyTorch ImageFolder...")
    try:
        from torchvision import datasets
        
        train_dataset = datasets.ImageFolder(str(train_dir))
        val_dataset = datasets.ImageFolder(str(val_dir))
        
        print(f"✓ Train dataset: {len(train_dataset)} imágenes, {len(train_dataset.classes)} clases")
        print(f"✓ Val dataset: {len(val_dataset)} imágenes, {len(val_dataset.classes)} clases")
        
        # Verificar que las clases coinciden
        if set(train_dataset.classes) == set(val_dataset.classes):
            print(f"✓ Las clases de train y val coinciden")
        else:
            print(f"⚠️  Las clases de train y val NO coinciden")
        
        # Probar cargar una imagen
        print()
        print("Probando carga de una imagen...")
        img, label = train_dataset[0]
        print(f"✓ Imagen cargada correctamente: {img.size}, label={label}")
        
    except ImportError:
        print("⚠️  torchvision no instalado, saltando verificación PyTorch")
    except Exception as e:
        print(f"❌ Error al cargar dataset con PyTorch: {e}")
    
    print()
    if checks_passed == total_checks:
        print("🎉 ¡Dataset verificado correctamente! Listo para entrenar.")
        print()
        print("Siguiente paso:")
        print("  sbatch train_var.slurm")
        return True
    else:
        print("⚠️  Algunas verificaciones fallaron. Revisa los errores arriba.")
        print()
        print("Si el dataset no está descargado:")
        print("  ./scripts/download_imagenet_localization.sh")
        return False

if __name__ == '__main__':
    # Determinar ruta del dataset
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        # Usar ruta por defecto
        script_dir = Path(__file__).parent.parent
        data_path = script_dir / 'data' / 'imagenet_localization'
    
    success = check_dataset(data_path)
    sys.exit(0 if success else 1)
