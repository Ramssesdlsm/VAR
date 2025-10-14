#!/usr/bin/env python3
"""
Script de verificación final antes de descargar ImageNet
"""

import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def check_status(name, status, details=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {name}")
    if details:
        for line in details.split('\n'):
            if line.strip():
                print(f"   {line}")
    return status


def main():
    print_header("Verificación Pre-Descarga de ImageNet")
    
    all_checks = []
    
    # 1. Verificar PATH
    print("\n1. Verificando Kaggle CLI...")
    kaggle_path = os.path.expanduser("~/.local/bin/kaggle")
    
    if os.path.exists(kaggle_path):
        # Agregar a PATH
        os.environ['PATH'] = f"{os.path.expanduser('~/.local/bin')}:{os.environ.get('PATH', '')}"
        
        try:
            result = subprocess.run(['kaggle', '--version'], 
                                  capture_output=True, text=True)
            version = result.stdout.strip()
            all_checks.append(check_status(
                "Kaggle CLI instalado",
                True,
                f"Ubicación: {kaggle_path}\n{version}"
            ))
        except Exception as e:
            all_checks.append(check_status(
                "Kaggle CLI instalado",
                False,
                f"Error al ejecutar: {str(e)}"
            ))
    else:
        all_checks.append(check_status(
            "Kaggle CLI instalado",
            False,
            f"No encontrado en: {kaggle_path}\nEjecuta: pip install --user kaggle"
        ))
    
    # 2. Verificar credenciales
    print("\n2. Verificando credenciales...")
    kaggle_config = Path.home() / '.kaggle' / 'kaggle.json'
    
    if kaggle_config.exists():
        perms = oct(kaggle_config.stat().st_mode)[-3:]
        status = perms == '600'
        all_checks.append(check_status(
            "Credenciales de Kaggle",
            True,
            f"Archivo: {kaggle_config}\nPermisos: {perms} {'✓' if status else '⚠️ Debería ser 600'}"
        ))
        
        if not status:
            print("   🔧 Corrigiendo permisos...")
            kaggle_config.chmod(0o600)
            print("   ✅ Permisos corregidos a 600")
    else:
        all_checks.append(check_status(
            "Credenciales de Kaggle",
            False,
            f"No encontrado: {kaggle_config}\n"
            "1. Ve a: https://www.kaggle.com/settings/account\n"
            "2. Click 'Create New Token'\n"
            "3. mv ~/Downloads/kaggle.json ~/.kaggle/\n"
            "4. chmod 600 ~/.kaggle/kaggle.json"
        ))
    
    # 3. Verificar acceso a competencia
    print("\n3. Verificando acceso a la competencia...")
    try:
        result = subprocess.run(
            ['kaggle', 'competitions', 'files', '-c', 
             'imagenet-object-localization-challenge'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            all_checks.append(check_status(
                "Acceso a competencia",
                True,
                "Reglas aceptadas - Listo para descargar"
            ))
        else:
            error_msg = result.stderr.lower()
            if '403' in error_msg or 'forbidden' in error_msg:
                all_checks.append(check_status(
                    "Acceso a competencia",
                    False,
                    "Debes aceptar las reglas:\n"
                    "https://www.kaggle.com/competitions/imagenet-object-localization-challenge\n"
                    "Click en 'Join Competition' o 'Late Submission'"
                ))
            else:
                all_checks.append(check_status(
                    "Acceso a competencia",
                    False,
                    f"Error: {result.stderr[:200]}"
                ))
    except subprocess.TimeoutExpired:
        all_checks.append(check_status(
            "Acceso a competencia",
            False,
            "Timeout - Verifica tu conexión a internet"
        ))
    except Exception as e:
        all_checks.append(check_status(
            "Acceso a competencia",
            False,
            f"Error al verificar: {str(e)}"
        ))
    
    # 4. Verificar espacio en disco
    print("\n4. Verificando espacio en disco...")
    project_root = Path(__file__).parent.parent
    stat = os.statvfs(project_root)
    available_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    required_gb = 200
    
    all_checks.append(check_status(
        "Espacio en disco",
        available_gb >= required_gb,
        f"Disponible: {available_gb:.1f} GB\n"
        f"Requerido: {required_gb} GB\n"
        f"Estado: {'✓ Suficiente' if available_gb >= required_gb else '⚠️ Insuficiente'}"
    ))
    
    # 5. Verificar screen
    print("\n5. Verificando screen/tmux...")
    screen_installed = subprocess.run(['which', 'screen'], 
                                     capture_output=True).returncode == 0
    tmux_installed = subprocess.run(['which', 'tmux'], 
                                   capture_output=True).returncode == 0
    
    if screen_installed or tmux_installed:
        tools = []
        if screen_installed:
            tools.append("screen")
        if tmux_installed:
            tools.append("tmux")
        all_checks.append(check_status(
            "Herramientas de sesión",
            True,
            f"Instalado: {', '.join(tools)}"
        ))
    else:
        all_checks.append(check_status(
            "Herramientas de sesión",
            False,
            "No instalados screen ni tmux\n"
            "Recomendado instalar: sudo apt-get install screen\n"
            "O usar: nohup bash scripts/download_imagenet_robust.sh &"
        ))
    
    # 6. Verificar directorios
    print("\n6. Verificando estructura de directorios...")
    kaggle_dir = project_root / 'kaggle'
    kaggle_dir.mkdir(exist_ok=True)
    
    all_checks.append(check_status(
        "Directorios del proyecto",
        True,
        f"Kaggle: {kaggle_dir}\n"
        f"Proyecto: {project_root}"
    ))
    
    # Resumen final
    print_header("Resumen")
    
    passed = sum(all_checks)
    total = len(all_checks)
    percentage = (passed / total) * 100
    
    print(f"\nPuntuación: {passed}/{total} ({percentage:.0f}%)")
    
    if passed == total:
        print_header("✅ TODO LISTO PARA DESCARGAR")
        print("\n🚀 Ejecuta uno de estos comandos:\n")
        print("Opción 1 (con screen - recomendado):")
        print("  bash scripts/start_download_with_screen.sh")
        print("\nOpción 2 (directo):")
        print("  bash scripts/download_imagenet_robust.sh")
        print("\nOpción 3 (con nohup):")
        print("  nohup bash scripts/download_imagenet_robust.sh > download.log 2>&1 &")
        print("")
        return 0
    elif passed >= total - 1:
        print_header("⚠️ CASI LISTO - Falta 1 cosa")
        print("\n🔧 Revisa los mensajes arriba y corrige el problema.")
        print("")
        return 1
    else:
        print_header("❌ NECESITA CONFIGURACIÓN")
        print(f"\n🔧 Faltan {total - passed} verificaciones.")
        print("   Revisa los mensajes arriba para solucionarlos.")
        print("")
        return 2


if __name__ == '__main__':
    sys.exit(main())
