#!/usr/bin/env python3
"""
Script de verificación pre-entrenamiento para VAR con ImageNet
Verifica que todo esté listo antes de iniciar el entrenamiento
"""

import os
import sys
from pathlib import Path

def print_header(title):
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")

def check_mark(condition, success_msg, fail_msg):
    if condition:
        print(f"✅ {success_msg}")
        return True
    else:
        print(f"❌ {fail_msg}")
        return False

def main():
    project_dir = Path("/home/est_posgrado_ramsses.delossantos/VAR")
    os.chdir(project_dir)
    
    all_checks_passed = True
    
    print_header("Verificación Pre-Entrenamiento VAR")
    
    # ========================================================================
    # 1. Verificar estructura del dataset
    # ========================================================================
    print_header("1. Estructura del Dataset")
    
    data_dir = project_dir / "data" / "imagenet"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    all_checks_passed &= check_mark(
        data_dir.exists(),
        f"Directorio data/imagenet existe",
        f"Directorio data/imagenet NO existe"
    )
    
    all_checks_passed &= check_mark(
        train_dir.exists(),
        f"Directorio train/ existe",
        f"Directorio train/ NO existe"
    )
    
    all_checks_passed &= check_mark(
        val_dir.exists(),
        f"Directorio val/ existe",
        f"Directorio val/ NO existe"
    )
    
    # Verificar enlaces simbólicos
    if train_dir.exists():
        is_symlink = train_dir.is_symlink()
        target = train_dir.resolve() if is_symlink else None
        
        all_checks_passed &= check_mark(
            is_symlink,
            f"train/ es enlace simbólico → {target}",
            f"train/ NO es enlace simbólico"
        )
        
        if is_symlink and target:
            all_checks_passed &= check_mark(
                target.exists(),
                f"Destino del enlace existe y es accesible",
                f"Destino del enlace NO existe o no es accesible"
            )
    
    # ========================================================================
    # 2. Contar clases e imágenes
    # ========================================================================
    print_header("2. Conteo de Clases e Imágenes")
    
    if train_dir.exists() and train_dir.is_dir():
        try:
            classes = [d for d in train_dir.iterdir() if d.is_dir()]
            num_classes = len(classes)
            
            all_checks_passed &= check_mark(
                num_classes == 1000,
                f"Train tiene {num_classes} clases (esperado: 1000)",
                f"Train tiene {num_classes} clases (esperado: 1000)"
            )
            
            # Contar imágenes en las primeras 3 clases (para no tardar mucho)
            print("\n  Verificando clases de ejemplo:")
            for class_dir in sorted(classes)[:3]:
                images = list(class_dir.glob("*.JPEG"))
                print(f"    {class_dir.name}: {len(images)} imágenes")
            
        except Exception as e:
            print(f"❌ Error al contar clases: {e}")
            all_checks_passed = False
    
    if val_dir.exists() and val_dir.is_dir():
        try:
            val_images = list(val_dir.glob("*.JPEG"))
            num_val = len(val_images)
            
            all_checks_passed &= check_mark(
                num_val == 50000,
                f"Val tiene {num_val:,} imágenes (esperado: 50,000)",
                f"Val tiene {num_val:,} imágenes (esperado: 50,000)"
            )
        except Exception as e:
            print(f"❌ Error al contar val: {e}")
            all_checks_passed = False
    
    # ========================================================================
    # 3. Verificar checkpoints VAE
    # ========================================================================
    print_header("3. Checkpoint VAE")
    
    vae_ckpt = project_dir / "vae_ch160v4096z32.pth"
    
    all_checks_passed &= check_mark(
        vae_ckpt.exists(),
        f"VAE checkpoint existe: {vae_ckpt.name} ({vae_ckpt.stat().st_size / 1024**2:.1f} MB)",
        f"VAE checkpoint NO existe: {vae_ckpt}"
    )
    
    # ========================================================================
    # 4. Verificar directorio de salida
    # ========================================================================
    print_header("4. Directorio de Salida")
    
    output_dir = project_dir / "output" / "var_d3"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_checks_passed &= check_mark(
        output_dir.exists(),
        f"Directorio output/var_d3 creado",
        f"No se pudo crear output/var_d3"
    )
    
    # Verificar checkpoints existentes
    checkpoints = list(output_dir.glob("checkpoint-*.pth"))
    if checkpoints:
        print(f"  ⚠️  Encontrados {len(checkpoints)} checkpoints existentes")
        print(f"      El entrenamiento se reanudará automáticamente (--auto_resume)")
        latest = sorted(checkpoints)[-1]
        print(f"      Último checkpoint: {latest.name}")
    else:
        print(f"  ℹ️  No hay checkpoints previos (entrenamiento desde cero)")
    
    # ========================================================================
    # 5. Verificar logs
    # ========================================================================
    print_header("5. Directorio de Logs")
    
    logs_dir = project_dir / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    all_checks_passed &= check_mark(
        logs_dir.exists(),
        f"Directorio logs/ existe",
        f"No se pudo crear logs/"
    )
    
    # ========================================================================
    # 6. Verificar script de entrenamiento
    # ========================================================================
    print_header("6. Script de Entrenamiento")
    
    train_script = project_dir / "train.py"
    slurm_script = project_dir / "train_var.slurm"
    
    all_checks_passed &= check_mark(
        train_script.exists(),
        f"Script train.py existe",
        f"Script train.py NO existe"
    )
    
    all_checks_passed &= check_mark(
        slurm_script.exists(),
        f"Script train_var.slurm existe",
        f"Script train_var.slurm NO existe"
    )
    
    # Verificar configuración en SLURM script
    if slurm_script.exists():
        with open(slurm_script, 'r') as f:
            content = f.read()
            
            has_correct_path = "data/imagenet" in content
            all_checks_passed &= check_mark(
                has_correct_path,
                f"SLURM script usa data_path correcto (data/imagenet)",
                f"SLURM script NO usa data_path correcto"
            )
    
    # ========================================================================
    # 7. Verificar Python y dependencias
    # ========================================================================
    print_header("7. Entorno Python")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            num_gpus = torch.cuda.device_count()
            print(f"✅ CUDA disponible: {num_gpus} GPU(s)")
            for i in range(num_gpus):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print(f"⚠️  CUDA NO disponible (solo CPU)")
        
    except ImportError:
        print(f"❌ PyTorch no está instalado")
        all_checks_passed = False
    
    try:
        import torchvision
        print(f"✅ torchvision {torchvision.__version__}")
    except ImportError:
        print(f"❌ torchvision no está instalado")
        all_checks_passed = False
    
    # ========================================================================
    # 8. Espacio en disco
    # ========================================================================
    print_header("8. Espacio en Disco")
    
    import shutil
    
    total, used, free = shutil.disk_usage(project_dir)
    free_gb = free // (2**30)
    
    all_checks_passed &= check_mark(
        free_gb > 50,
        f"Espacio disponible: {free_gb} GB (suficiente)",
        f"Espacio disponible: {free_gb} GB (puede ser insuficiente para checkpoints)"
    )
    
    # ========================================================================
    # Resumen final
    # ========================================================================
    print_header("RESUMEN")
    
    if all_checks_passed:
        print("🎉 ¡TODAS LAS VERIFICACIONES PASARON!")
        print("\n✅ El sistema está listo para entrenamiento\n")
        print("Para iniciar el entrenamiento:")
        print("    cd /home/est_posgrado_ramsses.delossantos/VAR")
        print("    sbatch train_var.slurm")
        print()
        return 0
    else:
        print("❌ ALGUNAS VERIFICACIONES FALLARON")
        print("\n⚠️  Por favor resuelve los problemas antes de entrenar")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
