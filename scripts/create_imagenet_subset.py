#!/usr/bin/env python3
"""
Script para crear un subset balanceado de ImageNet manteniendo la misma proporción
de imágenes por clase.

Uso:
    python3 scripts/create_imagenet_subset.py --ratio 0.5
    python3 scripts/create_imagenet_subset.py --ratio 0.25
"""

import os
import sys
import random
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm

def create_balanced_subset(source_dir, target_dir, ratio=None, images_per_class=None, seed=42, use_symlinks=True):
    """
    Crea un subset balanceado copiando o enlazando imágenes por clase.
    
    Args:
        source_dir: Directorio fuente (train o val)
        target_dir: Directorio destino
        ratio: Proporción a mantener (0.5 = 50%, 0.25 = 25%). Ignorado si images_per_class está definido.
        images_per_class: Número exacto de imágenes por clase. Tiene prioridad sobre ratio.
        seed: Semilla para reproducibilidad
        use_symlinks: Si True, usa enlaces simbólicos (rápido, ahorra espacio)
                      Si False, copia archivos (lento, duplica datos)
    """
    random.seed(seed)
    
    source_path = Path(source_dir).resolve()  # Resolver ruta absoluta
    target_path = Path(target_dir)
    
    if not source_path.exists():
        print(f"❌ ERROR: {source_path} no existe")
        return False
    
    # Crear directorio destino
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Obtener todas las clases (subdirectorios)
    classes = sorted([d for d in source_path.iterdir() if d.is_dir()])
    
    if len(classes) == 0:
        print(f"❌ ERROR: No se encontraron clases en {source_path}")
        return False
    
    print(f"Encontradas {len(classes)} clases")
    print(f"Método: {'Enlaces simbólicos (symlinks)' if use_symlinks else 'Copia de archivos'}")
    
    # Determinar modo de selección
    if images_per_class is not None:
        print(f"Modo: {images_per_class} imágenes por clase (fijo)")
    else:
        print(f"Modo: {ratio:.1%} de imágenes por clase (proporción)")
    
    total_images = 0
    total_copied = 0
    
    # Procesar cada clase
    for class_dir in tqdm(classes, desc=f"Procesando clases"):
        class_name = class_dir.name
        
        # Obtener todas las imágenes de esta clase
        images = list(class_dir.glob("*.JPEG"))
        
        if len(images) == 0:
            print(f"⚠️  Advertencia: No hay imágenes en {class_name}")
            continue
        
        # Calcular cuántas imágenes mantener
        if images_per_class is not None:
            # Modo: número fijo por clase
            num_to_keep = min(images_per_class, len(images))
            if len(images) < images_per_class:
                print(f"⚠️  {class_name}: solo {len(images)} imágenes (solicitadas {images_per_class})")
        else:
            # Modo: proporción
            num_to_keep = max(1, int(len(images) * ratio))
        
        # Seleccionar aleatoriamente
        selected_images = random.sample(images, num_to_keep)
        
        # Crear directorio de clase en destino
        target_class_dir = target_path / class_name
        target_class_dir.mkdir(exist_ok=True)
        
        # Crear symlinks o copiar imágenes seleccionadas
        for img in selected_images:
            target_img = target_class_dir / img.name
            
            if use_symlinks:
                # Crear enlace simbólico (instantáneo, no duplica datos)
                if not target_img.exists():
                    target_img.symlink_to(img)
            else:
                # Copiar archivo (lento, duplica datos)
                shutil.copy2(img, target_img)
        
        total_images += len(images)
        total_copied += num_to_keep
    
    print(f"\n✅ Completado:")
    print(f"   Original: {total_images:,} imágenes")
    print(f"   Subset: {total_copied:,} imágenes")
    print(f"   Promedio por clase: {total_copied/len(classes):.1f}")
    print(f"   Ratio real: {total_copied/total_images:.2%}")
    if use_symlinks:
        print(f"   💾 Espacio ahorrado: ~{(total_copied * 0.15):.1f} GB (usando symlinks)")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description='Crear subset balanceado de ImageNet',
        epilog='''
Ejemplos de uso:
  # Modo proporción (antiguo):
  python3 scripts/create_imagenet_subset.py --ratio 0.5
  
  # Modo número fijo (NUEVO):
  python3 scripts/create_imagenet_subset.py --train-images 200 --val-images 25
  python3 scripts/create_imagenet_subset.py --train-images 150 --val-images 50
        ''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Modo 1: Proporción (para compatibilidad)
    parser.add_argument('--ratio', type=float,
                        help='Proporción a mantener (0.5 = 50%%, 0.25 = 25%%). Ignorado si --train-images está definido.')
    
    # Modo 2: Número fijo por clase (RECOMENDADO)
    parser.add_argument('--train-images', type=int,
                        help='Número de imágenes por clase para TRAIN')
    parser.add_argument('--val-images', type=int,
                        help='Número de imágenes por clase para VAL')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla para reproducibilidad')
    parser.add_argument('--project-dir', type=str,
                        default='/home/est_posgrado_ramsses.delossantos/VAR',
                        help='Directorio del proyecto')
    parser.add_argument('--copy', action='store_true',
                        help='Copiar archivos en lugar de usar symlinks (más lento, usa más espacio)')
    
    args = parser.parse_args()
    
    use_symlinks = not args.copy
    
    # Validar argumentos
    if args.train_images is not None or args.val_images is not None:
        # Modo: número fijo
        if args.train_images is None or args.val_images is None:
            print("❌ ERROR: Debes especificar tanto --train-images como --val-images")
            return 1
        if args.train_images <= 0 or args.val_images <= 0:
            print("❌ ERROR: --train-images y --val-images deben ser > 0")
            return 1
        
        mode = "fixed"
        train_param = args.train_images
        val_param = args.val_images
        
    elif args.ratio is not None:
        # Modo: proporción (compatibilidad)
        if args.ratio <= 0 or args.ratio > 1:
            print("❌ ERROR: --ratio debe estar entre 0 y 1")
            return 1
        
        mode = "ratio"
        train_param = args.ratio
        val_param = args.ratio
        
    else:
        print("❌ ERROR: Debes especificar --train-images/--val-images O --ratio")
        print("\nEjemplos:")
        print("  python3 scripts/create_imagenet_subset.py --train-images 200 --val-images 25")
        print("  python3 scripts/create_imagenet_subset.py --ratio 0.5")
        return 1
    
    project_dir = Path(args.project_dir)
    
    # Directorios fuente
    source_train = project_dir / "kaggle" / "ILSVRC" / "Data" / "CLS-LOC" / "train"
    source_val = project_dir / "kaggle" / "ILSVRC" / "Data" / "CLS-LOC" / "val"
    
    # Directorios destino
    if mode == "fixed":
        target_base = project_dir / "data" / f"imagenet_{train_param}train_{val_param}val"
    else:
        ratio_str = str(int(train_param * 100))
        target_base = project_dir / "data" / f"imagenet_{ratio_str}pct"
    
    target_train = target_base / "train"
    target_val = target_base / "val"
    
    print("=" * 80)
    if mode == "fixed":
        print(f"  Creando Subset de ImageNet ({train_param} train / {val_param} val por clase)")
    else:
        print(f"  Creando Subset Balanceado de ImageNet ({int(train_param*100)}%)")
    print("=" * 80)
    print()
    
    if mode == "fixed":
        print(f"Train: {train_param} imágenes por clase")
        print(f"Val: {val_param} imágenes por clase")
    else:
        print(f"Ratio: {train_param:.1%}")
    
    print(f"Seed: {args.seed}")
    print(f"Método: {'Copia de archivos' if args.copy else 'Enlaces simbólicos (symlinks)'}")
    print()
    print(f"Fuente train: {source_train}")
    print(f"Fuente val: {source_val}")
    print()
    print(f"Destino train: {target_train}")
    print(f"Destino val: {target_val}")
    print()
    
    # Verificar si ya existe
    if target_base.exists():
        print(f"⚠️  El directorio {target_base} ya existe")
        response = input("¿Eliminar y recrear? (s/n): ")
        if response.lower() == 's':
            shutil.rmtree(target_base)
            print("✅ Directorio eliminado")
        else:
            print("Operación cancelada")
            return 0
    
    print()
    print("=" * 80)
    print("  Procesando Train")
    print("=" * 80)
    print()
    
    if mode == "fixed":
        if not create_balanced_subset(source_train, target_train, 
                                      images_per_class=train_param, seed=args.seed, use_symlinks=use_symlinks):
            return 1
    else:
        if not create_balanced_subset(source_train, target_train, 
                                      ratio=train_param, seed=args.seed, use_symlinks=use_symlinks):
            return 1
    
    print()
    print("=" * 80)
    print("  Procesando Val")
    print("=" * 80)
    print()
    
    if mode == "fixed":
        if not create_balanced_subset(source_val, target_val, 
                                      images_per_class=val_param, seed=args.seed, use_symlinks=use_symlinks):
            return 1
    else:
        if not create_balanced_subset(source_val, target_val, 
                                      ratio=val_param, seed=args.seed, use_symlinks=use_symlinks):
            return 1
    
    print()
    print("=" * 80)
    print("  RESUMEN FINAL")
    print("=" * 80)
    print()
    
    # Contar imágenes finales
    train_images = sum(1 for _ in target_train.rglob("*.JPEG"))
    val_images = sum(1 for _ in target_val.rglob("*.JPEG"))
    train_classes = len(list(target_train.iterdir()))
    val_classes = len(list(target_val.iterdir()))
    
    print(f"✅ Subset creado exitosamente:")
    print()
    print(f"Train:")
    print(f"  - Clases: {train_classes}")
    print(f"  - Imágenes: {train_images:,}")
    print(f"  - Promedio por clase: {train_images/train_classes:.0f}")
    print()
    print(f"Val:")
    print(f"  - Clases: {val_classes}")
    print(f"  - Imágenes: {val_images:,}")
    print(f"  - Promedio por clase: {val_images/val_classes:.0f}")
    print()
    print(f"Total: {train_images + val_images:,} imágenes")
    print()
    
    # Calcular tiempo estimado con configuración actual (2.8s/step, batch=64)
    steps_per_epoch = train_images // 64
    seconds_per_epoch = steps_per_epoch * 2.8
    hours_per_epoch = seconds_per_epoch / 3600
    days_for_100_epochs = (hours_per_epoch * 100) / 24
    
    print("⏱️  Estimación de Tiempo (con batch=64, 2.8s/step):")
    print(f"  - Steps por época: ~{steps_per_epoch:,}")
    print(f"  - Tiempo por época: ~{hours_per_epoch:.1f} horas")
    print(f"  - Total 100 épocas: ~{days_for_100_epochs:.1f} días")
    print()
    
    print("Para usar este subset, modifica train_var.slurm:")
    if mode == "fixed":
        print(f"  --data_path data/imagenet_{train_param}train_{val_param}val \\")
    else:
        print(f"  --data_path data/imagenet_{ratio_str}pct \\")
    print()
    print("=" * 80)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
