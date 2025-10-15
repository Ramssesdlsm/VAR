#!/bin/bash

################################################################################
# Script para descomprimir y organizar ImageNet para entrenamiento VAR
# 
# Estructura final:
#   kaggle/ILSVRC/              ← Backup sin modificar
#   data/imagenet/              ← Organizado para entrenamiento
################################################################################

set -e

PROJECT_DIR="/home/est_posgrado_ramsses.delossantos/VAR"
KAGGLE_DIR="$PROJECT_DIR/kaggle"
DATA_DIR="$PROJECT_DIR/data"
ZIP_FILE="$KAGGLE_DIR/imagenet-object-localization-challenge.zip"

echo "========================================================================"
echo "  Organización de ImageNet para VAR"
echo "========================================================================"
echo "Fecha: $(date)"
echo ""

################################################################################
# Verificar archivo ZIP
################################################################################

echo "Paso 1: Verificando archivo ZIP..."
echo ""

if [ ! -f "$ZIP_FILE" ]; then
    echo "❌ ERROR: No se encontró $ZIP_FILE"
    exit 1
fi

ZIP_SIZE=$(du -h "$ZIP_FILE" | cut -f1)
echo "✅ Archivo ZIP encontrado: $ZIP_SIZE"
echo ""

if ! file "$ZIP_FILE" | grep -q "Zip archive"; then
    echo "❌ ERROR: El archivo no es un ZIP válido"
    exit 1
fi

echo "✅ Archivo ZIP válido"
echo ""

################################################################################
# Verificar espacio disponible
################################################################################

echo "Paso 2: Verificando espacio en disco..."
echo ""

AVAILABLE_GB=$(df -BG "$KAGGLE_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
echo "💾 Espacio disponible: ${AVAILABLE_GB} GB"

if [ "$AVAILABLE_GB" -lt 200 ]; then
    echo "⚠️  Advertencia: Se recomienda al menos 200 GB libres"
    echo "   (156 GB para descomprimir + copia para data/)"
    read -p "¿Continuar de todos modos? (s/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "Cancelado"
        exit 1
    fi
fi
echo ""

################################################################################
# Descomprimir en kaggle/ (Backup)
################################################################################

echo "========================================================================"
echo "  Paso 3: Descomprimiendo en kaggle/ (Backup)"
echo "========================================================================"
echo ""

cd "$KAGGLE_DIR"

if [ -d "ILSVRC" ]; then
    echo "⚠️  El directorio ILSVRC ya existe en kaggle/"
    echo ""
    
    # Verificar si está completo
    if [ -d "ILSVRC/Data/CLS-LOC/train" ] && [ -d "ILSVRC/Data/CLS-LOC/val" ]; then
        TRAIN_COUNT=$(find ILSVRC/Data/CLS-LOC/train -name "*.JPEG" 2>/dev/null | wc -l)
        VAL_COUNT=$(find ILSVRC/Data/CLS-LOC/val -name "*.JPEG" 2>/dev/null | wc -l)
        
        echo "  Train: $TRAIN_COUNT imágenes"
        echo "  Val: $VAL_COUNT imágenes"
        echo ""
        
        if [ "$TRAIN_COUNT" -gt 1000000 ] && [ "$VAL_COUNT" -gt 40000 ]; then
            echo "✅ El dataset ya está descomprimido y parece completo"
            echo "   Saltando descompresión..."
            echo ""
        else
            echo "⚠️  El dataset parece incompleto"
            read -p "¿Eliminar y descomprimir de nuevo? (s/n): " -n 1 -r
            echo
            
            if [[ $REPLY =~ ^[Ss]$ ]]; then
                echo "Eliminando ILSVRC incompleto..."
                rm -rf ILSVRC
            else
                echo "Continuando con el dataset existente..."
            fi
        fi
    fi
fi

if [ ! -d "ILSVRC" ]; then
    echo "Descomprimiendo archivo ZIP..."
    echo "⏱️  Esto puede tardar 30-60 minutos"
    echo ""
    
    START_TIME=$(date +%s)
    
    # Descomprimir con barra de progreso
    unzip -q "$ZIP_FILE"
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    MINUTES=$((ELAPSED / 60))
    
    echo ""
    echo "✅ Descompresión completada en $MINUTES minutos"
    echo ""
else
    echo "✅ Usando dataset existente en kaggle/ILSVRC"
    echo ""
fi

################################################################################
# Verificar estructura en kaggle/
################################################################################

echo "Paso 4: Verificando estructura en kaggle/..."
echo ""

if [ ! -d "ILSVRC/Data/CLS-LOC" ]; then
    echo "❌ ERROR: No se encontró ILSVRC/Data/CLS-LOC/"
    echo ""
    echo "Estructura encontrada:"
    find ILSVRC -maxdepth 3 -type d
    exit 1
fi

echo "✅ Estructura correcta en kaggle/ILSVRC"
echo ""

# Contar imágenes
echo "Contando imágenes en kaggle/ILSVRC..."
if [ -d "ILSVRC/Data/CLS-LOC/train" ]; then
    TRAIN_COUNT=$(find ILSVRC/Data/CLS-LOC/train -name "*.JPEG" | wc -l)
    echo "  Train: $TRAIN_COUNT imágenes"
fi

if [ -d "ILSVRC/Data/CLS-LOC/val" ]; then
    VAL_COUNT=$(find ILSVRC/Data/CLS-LOC/val -name "*.JPEG" | wc -l)
    echo "  Val: $VAL_COUNT imágenes"
fi

if [ -d "ILSVRC/Data/CLS-LOC/test" ]; then
    TEST_COUNT=$(find ILSVRC/Data/CLS-LOC/test -name "*.JPEG" | wc -l)
    echo "  Test: $TEST_COUNT imágenes"
fi

echo ""
KAGGLE_SIZE=$(du -sh ILSVRC | cut -f1)
echo "Tamaño total en kaggle/: $KAGGLE_SIZE"
echo ""

################################################################################
# Organizar en data/ para entrenamiento
################################################################################

echo "========================================================================"
echo "  Paso 5: Organizando en data/ para entrenamiento"
echo "========================================================================"
echo ""

mkdir -p "$DATA_DIR/imagenet"
cd "$DATA_DIR/imagenet"

if [ -d "train" ] || [ -L "train" ]; then
    echo "⚠️  Ya existe train/ en data/imagenet/"
    read -p "¿Eliminar y recrear? (s/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        rm -rf train val test
        echo "✅ Directorios anteriores eliminados"
    else
        echo "Manteniendo estructura existente"
    fi
fi

# Crear enlaces simbólicos (ahorra espacio)
if [ ! -e "train" ]; then
    echo "Creando enlaces simbólicos..."
    echo "  ├─ train/ → kaggle/ILSVRC/Data/CLS-LOC/train/"
    ln -s "$KAGGLE_DIR/ILSVRC/Data/CLS-LOC/train" train
    
    echo "  ├─ val/ → kaggle/ILSVRC/Data/CLS-LOC/val/"
    ln -s "$KAGGLE_DIR/ILSVRC/Data/CLS-LOC/val" val
    
    if [ -d "$KAGGLE_DIR/ILSVRC/Data/CLS-LOC/test" ]; then
        echo "  └─ test/ → kaggle/ILSVRC/Data/CLS-LOC/test/"
        ln -s "$KAGGLE_DIR/ILSVRC/Data/CLS-LOC/test" test
    fi
    
    echo ""
    echo "✅ Enlaces simbólicos creados"
else
    echo "✅ Enlaces ya existen"
fi

echo ""

################################################################################
# Verificar estructura final
################################################################################

echo "========================================================================"
echo "  Paso 6: Verificación Final"
echo "========================================================================"
echo ""

echo "📁 Estructura de directorios:"
echo ""
echo "kaggle/ILSVRC/Data/CLS-LOC/        ← Backup completo (sin modificar)"
ls -lh "$KAGGLE_DIR/ILSVRC/Data/CLS-LOC/" 2>/dev/null | head -10
echo ""

echo "data/imagenet/                      ← Para entrenamiento (enlaces simbólicos)"
ls -lh "$DATA_DIR/imagenet/" 2>/dev/null
echo ""

# Verificar que los enlaces funcionan
echo "Verificando enlaces simbólicos..."
if [ -L "$DATA_DIR/imagenet/train" ] && [ -e "$DATA_DIR/imagenet/train" ]; then
    TRAIN_LINK=$(readlink -f "$DATA_DIR/imagenet/train")
    TRAIN_FILES=$(ls "$DATA_DIR/imagenet/train" | head -5)
    echo "✅ train/ → $TRAIN_LINK"
    echo "   Primeras carpetas:"
    for folder in $(ls "$DATA_DIR/imagenet/train" | head -3); do
        COUNT=$(ls "$DATA_DIR/imagenet/train/$folder" 2>/dev/null | wc -l)
        echo "     - $folder/ ($COUNT imágenes)"
    done
else
    echo "❌ ERROR: train/ no funciona correctamente"
fi
echo ""

if [ -L "$DATA_DIR/imagenet/val" ] && [ -e "$DATA_DIR/imagenet/val" ]; then
    VAL_LINK=$(readlink -f "$DATA_DIR/imagenet/val")
    VAL_FILES=$(ls "$DATA_DIR/imagenet/val" 2>/dev/null | wc -l)
    echo "✅ val/ → $VAL_LINK"
    echo "   Total archivos: $VAL_FILES"
else
    echo "❌ ERROR: val/ no funciona correctamente"
fi
echo ""

################################################################################
# Información de espacio
################################################################################

echo "💾 Uso de espacio:"
echo ""
echo "Backup en kaggle/:"
du -sh "$KAGGLE_DIR/ILSVRC" 2>/dev/null || echo "  (calculando...)"

echo ""
echo "Dataset en data/ (enlaces simbólicos):"
du -sh "$DATA_DIR/imagenet" 2>/dev/null || echo "  ~0 MB (solo enlaces)"

echo ""
echo "Espacio disponible:"
df -h "$DATA_DIR" | grep -v Filesystem

################################################################################
# Resumen final
################################################################################

echo ""
echo "========================================================================"
echo "  ✅ ORGANIZACIÓN COMPLETADA"
echo "========================================================================"
echo ""
echo "📁 Estructura final:"
echo ""
echo "   kaggle/ILSVRC/Data/CLS-LOC/         ← BACKUP (sin modificar)"
echo "   ├─ train/                            (~1.28M imágenes)"
echo "   ├─ val/                              (~50K imágenes)"
echo "   └─ test/                             (~100K imágenes)"
echo ""
echo "   data/imagenet/                       ← PARA ENTRENAMIENTO"
echo "   ├─ train/  → kaggle/ILSVRC/.../train/"
echo "   ├─ val/    → kaggle/ILSVRC/.../val/"
echo "   └─ test/   → kaggle/ILSVRC/.../test/"
echo ""
echo "💡 Ventajas:"
echo "   ✅ Backup completo en kaggle/ (sin modificar)"
echo "   ✅ Enlaces simbólicos ahorran espacio (no duplican datos)"
echo "   ✅ Estructura lista para entrenamiento en data/imagenet/"
echo ""
echo "🚀 Próximos pasos:"
echo ""
echo "1. Verificar estructura:"
echo "   ls -lh $DATA_DIR/imagenet/"
echo ""
echo "2. Entrenar VAR:"
echo "   cd $PROJECT_DIR"
echo "   sbatch train_var.slurm"
echo ""
echo "========================================================================"
echo "Fin: $(date)"
echo "========================================================================"

exit 0
