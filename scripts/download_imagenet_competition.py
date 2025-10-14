#!/usr/bin/env python3
"""
Script para descargar ImageNet Object Localization Challenge desde Kaggle
Dataset oficial con ~150K imágenes
Fuente: https://www.kaggle.com/competitions/imagenet-object-localization-challenge
"""

import os
import sys
import subprocess
import tarfile
import shutil
from pathlib import Path


def print_header(text):
    """Imprime un encabezado formateado."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_status(message, symbol="→"):
    """Imprime un mensaje de estado."""
    print(f"{symbol} {message}")


def check_kaggle_cli():
    """Verifica e instala Kaggle CLI si es necesario."""
    try:
        result = subprocess.run(['kaggle', '--version'], 
                              capture_output=True, 
                              text=True)
        print_status(f"Kaggle CLI encontrado: {result.stdout.strip()}", "✅")
        return True
    except FileNotFoundError:
        print_status("Kaggle CLI no encontrado, instalando...", "⚠️")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'kaggle', '--user'],
                          check=True)
            print_status("Kaggle CLI instalado correctamente", "✅")
            return True
        except subprocess.CalledProcessError as e:
            print_status(f"Error al instalar Kaggle CLI: {e}", "❌")
            return False


def check_kaggle_credentials():
    """Verifica credenciales de Kaggle."""
    kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
    
    if not kaggle_config.exists():
        print_status("Credenciales de Kaggle no encontradas", "❌")
        print("\nPor favor, configura tus credenciales:")
        print("1. Ve a https://www.kaggle.com/settings/account")
        print("2. En la sección 'API', haz clic en 'Create New Token'")
        print("3. Ejecuta:")
        print("   mkdir -p ~/.kaggle")
        print("   mv ~/Downloads/kaggle.json ~/.kaggle/")
        print("   chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Verificar permisos
    perms = oct(kaggle_config.stat().st_mode)[-3:]
    if perms != '600':
        print_status(f"Corrigiendo permisos de kaggle.json ({perms} -> 600)", "⚠️")
        kaggle_config.chmod(0o600)
    
    print_status("Credenciales de Kaggle encontradas", "✅")
    return True


def accept_competition_rules():
    """Informa al usuario sobre aceptar las reglas de la competencia."""
    print_header("IMPORTANTE: Aceptar Reglas de la Competencia")
    print("\nAntes de descargar, debes aceptar las reglas de la competencia:")
    print("1. Ve a: https://www.kaggle.com/competitions/imagenet-object-localization-challenge")
    print("2. Haz clic en 'Join Competition' o 'Late Submission'")
    print("3. Acepta las reglas")
    print()
    
    response = input("¿Ya aceptaste las reglas? (y/N): ").strip().lower()
    if response != 'y':
        print_status("Por favor, acepta las reglas primero y vuelve a ejecutar este script", "⚠️")
        return False
    
    return True


def download_competition_data(kaggle_dir: Path):
    """Descarga los datos de la competencia."""
    print_header("Descargando ImageNet Object Localization Challenge")
    
    print_status("Esto descargará ~170 GB de datos (comprimidos)", "ℹ️")
    print_status("Tiempo estimado: 1-3 horas dependiendo de tu conexión", "ℹ️")
    print()
    
    response = input("¿Deseas continuar? (y/N): ").strip().lower()
    if response != 'y':
        print_status("Descarga cancelada", "⚠️")
        return False
    
    # Cambiar al directorio de descarga
    original_dir = Path.cwd()
    os.chdir(kaggle_dir)
    
    try:
        print()
        print_status("Descargando archivos de la competencia...", "📥")
        print_status("Este proceso puede tardar varias horas", "⏳")
        print()
        
        # Descargar todos los archivos de la competencia
        result = subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', 'imagenet-object-localization-challenge'],
            text=True
        )
        
        if result.returncode != 0:
            print_status("Error al descargar el dataset", "❌")
            print("\nPosibles soluciones:")
            print("1. Asegúrate de haber aceptado las reglas en:")
            print("   https://www.kaggle.com/competitions/imagenet-object-localization-challenge")
            print("2. Verifica tu conexión a internet")
            print("3. Verifica que tus credenciales sean válidas")
            return False
        
        print()
        print_status("Descarga completada", "✅")
        return True
        
    except Exception as e:
        print_status(f"Error durante la descarga: {e}", "❌")
        return False
    finally:
        os.chdir(original_dir)


def extract_archives(kaggle_dir: Path, imagenet_dir: Path):
    """Extrae los archivos descargados."""
    print_header("Extrayendo Archivos")
    
    # Buscar archivos .tar.gz
    archives = list(kaggle_dir.glob('*.tar.gz')) + list(kaggle_dir.glob('*.tar'))
    
    if not archives:
        print_status("No se encontraron archivos para extraer", "⚠️")
        return False
    
    print_status(f"Encontrados {len(archives)} archivos para extraer", "ℹ️")
    
    for archive in archives:
        print()
        print_status(f"Extrayendo {archive.name}...", "📦")
        
        try:
            # Determinar el directorio de destino
            if 'train' in archive.name.lower():
                dest_dir = imagenet_dir / 'train'
            elif 'val' in archive.name.lower():
                dest_dir = imagenet_dir / 'val'
            elif 'test' in archive.name.lower():
                dest_dir = imagenet_dir / 'test'
            else:
                dest_dir = imagenet_dir
            
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            # Extraer
            with tarfile.open(archive, 'r:*') as tar:
                tar.extractall(dest_dir)
            
            print_status(f"✓ {archive.name} extraído", "✅")
            
        except Exception as e:
            print_status(f"Error al extraer {archive.name}: {e}", "❌")
            return False
    
    return True


def organize_dataset(imagenet_dir: Path):
    """Organiza el dataset en la estructura correcta."""
    print_header("Organizando Dataset")
    
    # El dataset de ImageNet Object Localization viene con una estructura específica
    # Necesitamos verificar y ajustar si es necesario
    
    train_dir = imagenet_dir / 'ILSVRC' / 'Data' / 'CLS-LOC' / 'train'
    val_dir = imagenet_dir / 'ILSVRC' / 'Data' / 'CLS-LOC' / 'val'
    
    target_train = imagenet_dir / 'train'
    target_val = imagenet_dir / 'val'
    
    # Mover train si existe en la estructura anidada
    if train_dir.exists() and not target_train.exists():
        print_status("Moviendo directorio train a la raíz...", "📁")
        shutil.move(str(train_dir), str(target_train))
        print_status("✓ Train movido", "✅")
    
    # Mover val si existe en la estructura anidada
    if val_dir.exists() and not target_val.exists():
        print_status("Moviendo directorio val a la raíz...", "📁")
        shutil.move(str(val_dir), str(target_val))
        print_status("✓ Val movido", "✅")
    
    # Limpiar estructura ILSVRC si está vacía
    ilsvrc_dir = imagenet_dir / 'ILSVRC'
    if ilsvrc_dir.exists():
        try:
            # Verificar si hay algo más importante
            data_dir = ilsvrc_dir / 'Data'
            if data_dir.exists() and not any(data_dir.rglob('*.JPEG')):
                print_status("Limpiando estructura ILSVRC vacía...", "🧹")
                shutil.rmtree(ilsvrc_dir)
        except Exception:
            pass
    
    return True


def verify_dataset(imagenet_dir: Path):
    """Verifica la estructura y cuenta las imágenes del dataset."""
    print_header("Verificando Dataset")
    
    train_dir = imagenet_dir / 'train'
    val_dir = imagenet_dir / 'val'
    
    stats = {
        'train_images': 0,
        'val_images': 0,
        'train_classes': 0,
        'val_classes': 0,
    }
    
    # Verificar train
    if train_dir.exists():
        classes = [d for d in train_dir.iterdir() if d.is_dir()]
        stats['train_classes'] = len(classes)
        
        print_status(f"Contando imágenes de train ({stats['train_classes']} clases)...", "🔍")
        for class_dir in classes:
            images = list(class_dir.glob('*.JPEG')) + \
                    list(class_dir.glob('*.jpg')) + \
                    list(class_dir.glob('*.png'))
            stats['train_images'] += len(images)
        
        print_status(f"Train: {stats['train_images']:,} imágenes en {stats['train_classes']} clases", "✅")
    else:
        print_status("Directorio train no encontrado", "⚠️")
    
    # Verificar val
    if val_dir.exists():
        # Verificar si está organizado por clases o si son archivos sueltos
        val_subdirs = [d for d in val_dir.iterdir() if d.is_dir()]
        
        if val_subdirs:
            # Organizado por clases
            stats['val_classes'] = len(val_subdirs)
            print_status(f"Contando imágenes de val ({stats['val_classes']} clases)...", "🔍")
            for class_dir in val_subdirs:
                images = list(class_dir.glob('*.JPEG')) + \
                        list(class_dir.glob('*.jpg')) + \
                        list(class_dir.glob('*.png'))
                stats['val_images'] += len(images)
        else:
            # Archivos sueltos
            print_status("Contando imágenes de val (sin organizar por clases)...", "🔍")
            images = list(val_dir.glob('*.JPEG')) + \
                    list(val_dir.glob('*.jpg')) + \
                    list(val_dir.glob('*.png'))
            stats['val_images'] = len(images)
        
        print_status(f"Val: {stats['val_images']:,} imágenes", "✅")
    else:
        print_status("Directorio val no encontrado", "ℹ️")
    
    total_images = stats['train_images'] + stats['val_images']
    print()
    print_status(f"TOTAL: {total_images:,} imágenes", "📊")
    
    return stats


def cleanup_temp_files(kaggle_dir: Path):
    """Limpia archivos temporales."""
    print_header("Limpiando Archivos Temporales")
    
    response = input("\n¿Deseas eliminar los archivos .tar.gz descargados? (y/N): ").strip().lower()
    
    if response == 'y':
        removed = 0
        for pattern in ['*.tar.gz', '*.tar', '*.zip']:
            for file in kaggle_dir.glob(pattern):
                try:
                    file.unlink()
                    print_status(f"Eliminado: {file.name}", "🗑️")
                    removed += 1
                except Exception as e:
                    print_status(f"No se pudo eliminar {file.name}: {e}", "⚠️")
        
        if removed > 0:
            print_status(f"{removed} archivo(s) eliminado(s)", "✅")
        else:
            print_status("No hay archivos para eliminar", "ℹ️")
    else:
        print_status("Archivos temporales conservados", "ℹ️")


def main():
    """Función principal."""
    print_header("ImageNet Object Localization Challenge Downloader")
    print()
    print("Este script descargará el dataset oficial de ImageNet desde Kaggle")
    print("Competencia: imagenet-object-localization-challenge")
    print()
    
    # Configurar directorios
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / 'data'
    kaggle_dir = project_root / 'kaggle'
    imagenet_dir = data_dir / 'imagenet_localization'
    
    print_status(f"Proyecto: {project_root}", "📂")
    print_status(f"Destino: {imagenet_dir}", "📂")
    print()
    
    # Crear directorios
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    imagenet_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar dataset existente
    if (imagenet_dir / 'train').exists():
        print_status(f"Dataset ya existe en: {imagenet_dir}", "⚠️")
        response = input("\n¿Deseas re-descargar? (y/N): ").strip().lower()
        if response != 'y':
            print_status("Operación cancelada", "ℹ️")
            
            # Ofrecer solo verificar
            response = input("¿Deseas verificar el dataset existente? (y/N): ").strip().lower()
            if response == 'y':
                verify_dataset(imagenet_dir)
            
            return 0
        
        print_status("Eliminando dataset existente...", "🗑️")
        shutil.rmtree(imagenet_dir)
        imagenet_dir.mkdir(parents=True, exist_ok=True)
    
    # Paso 1: Verificar Kaggle CLI
    if not check_kaggle_cli():
        return 1
    
    # Paso 2: Verificar credenciales
    if not check_kaggle_credentials():
        return 1
    
    # Paso 3: Aceptar reglas
    if not accept_competition_rules():
        return 1
    
    # Paso 4: Descargar
    if not download_competition_data(kaggle_dir):
        return 1
    
    # Paso 5: Extraer
    if not extract_archives(kaggle_dir, imagenet_dir):
        return 1
    
    # Paso 6: Organizar
    if not organize_dataset(imagenet_dir):
        return 1
    
    # Paso 7: Verificar
    stats = verify_dataset(imagenet_dir)
    
    # Paso 8: Limpiar
    cleanup_temp_files(kaggle_dir)
    
    # Mensaje final
    print_header("✅ ¡Descarga Completada!")
    print()
    print_status(f"Dataset ubicado en: {imagenet_dir}", "📍")
    print()
    print("Para entrenar VAR:")
    print(f"  python train.py --data_path {imagenet_dir} \\")
    print("                  --epochs 250 \\")
    print("                  --batch_size 64")
    print()
    print("O usa SLURM:")
    print("  sbatch train_var_150k.slurm")
    print()
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
