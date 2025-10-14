#!/usr/bin/env python3
"""
Script para organizar el conjunto de validación de ImageNet en carpetas por clase.
ImageNet val viene con todas las imágenes en un solo directorio, pero necesitamos
organizarlas en subdirectorios por clase para que DatasetFolder funcione.
"""

import os
import csv
import shutil
from pathlib import Path
from tqdm import tqdm

def organize_val_dataset():
    """Organiza las imágenes de validación en subdirectorios por clase."""
    
    project_dir = Path("/home/est_posgrado_ramsses.delossantos/VAR")
    
    # Rutas
    val_dir = project_dir / "kaggle" / "ILSVRC" / "Data" / "CLS-LOC" / "val"
    val_solution = project_dir / "kaggle" / "LOC_val_solution.csv"
    
    print("=" * 80)
    print("  Organizando conjunto de validación por clases")
    print("=" * 80)
    print()
    
    # Verificar que existe el directorio val
    if not val_dir.exists():
        print(f"❌ ERROR: No se encontró {val_dir}")
        return False
    
    # Verificar que existe el archivo de soluciones
    if not val_solution.exists():
        print(f"❌ ERROR: No se encontró {val_solution}")
        return False
    
    print(f"✅ Directorio val: {val_dir}")
    print(f"✅ Archivo de soluciones: {val_solution}")
    print()
    
    # Verificar si ya está organizado
    subdirs = [d for d in val_dir.iterdir() if d.is_dir()]
    if len(subdirs) > 10:  # Si ya hay muchos subdirectorios, probablemente ya está organizado
        print(f"⚠️  El directorio val ya tiene {len(subdirs)} subdirectorios")
        print("   Parece que ya está organizado por clases")
        response = input("¿Reorganizar de todos modos? (s/n): ")
        if response.lower() != 's':
            print("Operación cancelada")
            return True
    
    # Leer el archivo CSV para obtener el mapeo imagen -> clase
    print("Leyendo archivo de soluciones...")
    image_to_class = {}
    
    with open(val_solution, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row['ImageId']
            prediction = row['PredictionString']
            # La clase es el primer elemento del prediction string
            class_id = prediction.split()[0]
            image_to_class[image_id] = class_id
    
    print(f"✅ Leídos {len(image_to_class)} mapeos imagen->clase")
    print()
    
    # Obtener lista de todas las imágenes JPEG en val
    print("Buscando imágenes en val/...")
    val_images = list(val_dir.glob("*.JPEG"))
    print(f"✅ Encontradas {len(val_images)} imágenes")
    print()
    
    if len(val_images) == 0:
        print("❌ ERROR: No se encontraron imágenes .JPEG en val/")
        print("   Puede que ya estén organizadas en subdirectorios")
        return False
    
    # Crear directorios para cada clase
    print("Creando directorios de clases...")
    unique_classes = set(image_to_class.values())
    print(f"Clases únicas: {len(unique_classes)}")
    
    for class_id in unique_classes:
        class_dir = val_dir / class_id
        class_dir.mkdir(exist_ok=True)
    
    print(f"✅ Creados {len(unique_classes)} directorios de clases")
    print()
    
    # Mover imágenes a sus respectivas carpetas
    print("Moviendo imágenes a subdirectorios por clase...")
    print("⏱️  Esto puede tardar unos minutos...")
    print()
    
    moved_count = 0
    error_count = 0
    
    for img_path in tqdm(val_images, desc="Organizando imágenes"):
        img_name = img_path.stem  # nombre sin extensión
        
        # Buscar la clase de esta imagen
        if img_name in image_to_class:
            class_id = image_to_class[img_name]
            dest_dir = val_dir / class_id
            dest_path = dest_dir / img_path.name
            
            try:
                # Mover la imagen
                shutil.move(str(img_path), str(dest_path))
                moved_count += 1
            except Exception as e:
                print(f"❌ Error moviendo {img_path.name}: {e}")
                error_count += 1
        else:
            print(f"⚠️  No se encontró clase para {img_name}")
            error_count += 1
    
    print()
    print("=" * 80)
    print("  RESUMEN")
    print("=" * 80)
    print()
    print(f"✅ Imágenes movidas: {moved_count:,}")
    if error_count > 0:
        print(f"⚠️  Errores: {error_count}")
    print()
    
    # Verificar estructura final
    print("Verificando estructura final...")
    class_dirs = [d for d in val_dir.iterdir() if d.is_dir()]
    print(f"✅ Directorios de clases: {len(class_dirs)}")
    
    # Contar imágenes en algunos directorios de ejemplo
    print()
    print("Ejemplos de directorios:")
    for class_dir in sorted(class_dirs)[:5]:
        images_in_class = list(class_dir.glob("*.JPEG"))
        print(f"  {class_dir.name}: {len(images_in_class)} imágenes")
    
    print()
    print("=" * 80)
    print("  ✅ ORGANIZACIÓN COMPLETADA")
    print("=" * 80)
    print()
    print("El conjunto de validación ahora está organizado por clases")
    print("Puedes proceder con el entrenamiento")
    print()
    
    return True

if __name__ == "__main__":
    import sys
    success = organize_val_dataset()
    sys.exit(0 if success else 1)
