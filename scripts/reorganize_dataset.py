#!/usr/bin/env python3
"""
Script para reorganizar el dataset ImageNet Localization con split 80/20 correcto.
Train: 80% de las imágenes (40,000)
Val: 20% de las imágenes (10,000)
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

def reorganize_dataset():
    """Reorganiza el dataset con split 80/20 correcto."""
    
    base_dir = Path('/home/est_posgrado_ramsses.delossantos/VAR/data/imagenet_localization')
    val_dir = base_dir / 'val'
    train_dir = base_dir / 'train'
    
    print("=" * 60)
    print("Reorganización de Dataset ImageNet Localization")
    print("=" * 60)
    print()
    
    # Verificar que existen los directorios
    if not val_dir.exists():
        print(f"Error: {val_dir} no existe")
        return False
    
    print("[1/5] Analizando dataset actual...")
    
    # Contar imágenes actuales
    train_images_before = list(train_dir.rglob('*.JPEG'))
    val_images_before = list(val_dir.rglob('*.JPEG'))
    
    print(f"  Train actual: {len(train_images_before)} imágenes")
    print(f"  Val actual: {len(val_images_before)} imágenes")
    print(f"  Total: {len(train_images_before) + len(val_images_before)} imágenes")
    
    # El problema: train tiene symlinks, val tiene las imágenes reales
    # Solución: eliminar train, reorganizar desde val
    
    print()
    print("[2/5] Respaldando estructura...")
    
    # Obtener todas las clases desde val (que tiene las imágenes reales)
    classes = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
    print(f"  {len(classes)} clases encontradas")
    
    if len(classes) != 1000:
        print(f"  ⚠️ Advertencia: Se esperaban 1000 clases, se encontraron {len(classes)}")
    
    print()
    print("[3/5] Eliminando train actual (symlinks)...")
    
    if train_dir.exists():
        shutil.rmtree(train_dir)
        print(f"  ✓ {train_dir} eliminado")
    
    train_dir.mkdir(exist_ok=True)
    
    print()
    print("[4/5] Reorganizando con split 80/20...")
    
    random.seed(42)  # Para reproducibilidad
    
    total_train = 0
    total_val_new = 0
    
    for class_name in classes:
        val_class_dir = val_dir / class_name
        train_class_dir = train_dir / class_name
        
        # Crear directorio de clase en train
        train_class_dir.mkdir(exist_ok=True)
        
        # Obtener todas las imágenes de esta clase
        images = sorted([f for f in val_class_dir.iterdir() if f.suffix == '.JPEG'])
        
        if len(images) == 0:
            print(f"  ⚠️ Clase {class_name} vacía, saltando...")
            continue
        
        # Mezclar y dividir 80/20
        random.shuffle(images)
        split_idx = int(0.8 * len(images))
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Mover imágenes a train
        for img in train_images:
            dest = train_class_dir / img.name
            shutil.move(str(img), str(dest))
        
        total_train += len(train_images)
        total_val_new += len(val_images)
        
        if (classes.index(class_name) + 1) % 100 == 0:
            print(f"  Procesadas {classes.index(class_name) + 1}/{len(classes)} clases...")
    
    print()
    print("[5/5] Limpiando carpetas vacías...")
    
    # Eliminar carpetas vacías en val
    for class_dir in val_dir.iterdir():
        if class_dir.is_dir():
            images_remaining = list(class_dir.glob('*.JPEG'))
            if len(images_remaining) == 0:
                class_dir.rmdir()
    
    print()
    print("=" * 60)
    print("Reorganización Completada")
    print("=" * 60)
    print()
    print(f"Resultado:")
    print(f"  Train: {total_train} imágenes ({total_train/(total_train+total_val_new)*100:.1f}%)")
    print(f"  Val:   {total_val_new} imágenes ({total_val_new/(total_train+total_val_new)*100:.1f}%)")
    print(f"  Total: {total_train + total_val_new} imágenes")
    print()
    
    # Verificación final
    print("Verificación final:")
    train_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
    val_classes = len([d for d in val_dir.iterdir() if d.is_dir()])
    print(f"  Train: {train_classes} clases")
    print(f"  Val:   {val_classes} clases")
    
    if train_classes == val_classes == 1000:
        print()
        print("✓ Dataset reorganizado correctamente con split 80/20")
        return True
    else:
        print()
        print("⚠️ Advertencia: Número de clases inesperado")
        return False

if __name__ == '__main__':
    import sys
    
    print()
    print("Este script reorganizará el dataset con split 80/20:")
    print("  - Train: 80% (~40,000 imágenes)")
    print("  - Val:   20% (~10,000 imágenes)")
    print()
    print("ADVERTENCIA: Esto modificará la estructura del dataset.")
    print("             Los symlinks actuales en train/ se eliminarán.")
    print()
    
    response = input("¿Continuar? (s/n): ")
    
    if response.lower() in ['s', 'si', 'y', 'yes']:
        success = reorganize_dataset()
        sys.exit(0 if success else 1)
    else:
        print("Operación cancelada.")
        sys.exit(0)
