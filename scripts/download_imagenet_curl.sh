#!/bin/bash

################################################################################
# Script para descargar ImageNet usando curl directamente
# Se ejecuta en el nodo de login (tiene acceso a internet)
################################################################################

set -e

echo "========================================================================"
echo "  ImageNet Download usando curl"
echo "========================================================================"
echo "Fecha: $(date)"
echo ""

# Configuración
KAGGLE_CONFIG="$HOME/.kaggle/kaggle.json"
PROJECT_DIR="/home/est_posgrado_ramsses.delossantos/VAR"
KAGGLE_DIR="$PROJECT_DIR/kaggle"
COMPETITION="imagenet-object-localization-challenge"
OUTPUT_FILE="$KAGGLE_DIR/imagenet-object-localization-challenge.zip"

# Crear directorio si no existe
mkdir -p "$KAGGLE_DIR"
cd "$KAGGLE_DIR"

################################################################################
# Verificar credenciales
################################################################################

echo "Verificando credenciales de Kaggle..."

if [ ! -f "$KAGGLE_CONFIG" ]; then
    echo "❌ ERROR: No se encontró $KAGGLE_CONFIG"
    echo ""
    echo "Crea el archivo con tus credenciales:"
    echo '  {"username":"TU_USERNAME","key":"TU_API_KEY"}'
    exit 1
fi

# Leer credenciales
KAGGLE_USERNAME=$(python3 -c "import json; print(json.load(open('$KAGGLE_CONFIG'))['username'])")
KAGGLE_KEY=$(python3 -c "import json; print(json.load(open('$KAGGLE_CONFIG'))['key'])")

if [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo "❌ ERROR: Credenciales inválidas en $KAGGLE_CONFIG"
    exit 1
fi

echo "✅ Usuario: $KAGGLE_USERNAME"
echo ""

################################################################################
# Verificar espacio disponible
################################################################################

echo "Verificando espacio en disco..."
AVAILABLE_GB=$(df -BG "$KAGGLE_DIR" | awk 'NR==2 {print $4}' | sed 's/G//')
echo "💾 Espacio disponible: ${AVAILABLE_GB} GB"

if [ "$AVAILABLE_GB" -lt 200 ]; then
    echo "⚠️  Advertencia: Se necesitan al menos 200 GB"
    echo "   (170 GB para ZIP + espacio para descomprimir)"
    read -p "¿Continuar de todos modos? (s/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Ss]$ ]]; then
        echo "Cancelado por el usuario"
        exit 1
    fi
fi
echo ""

################################################################################
# Verificar si ya existe el archivo
################################################################################

if [ -f "$OUTPUT_FILE" ]; then
    FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
    echo "⚠️  El archivo ya existe: $OUTPUT_FILE"
    echo "   Tamaño actual: $FILE_SIZE"
    echo ""
    read -p "¿Eliminar y descargar de nuevo? (s/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo "Eliminando archivo existente..."
        rm "$OUTPUT_FILE"
    else
        echo "Manteniendo archivo existente. Saliendo..."
        exit 0
    fi
fi

################################################################################
# Descargar con curl
################################################################################

echo "========================================================================"
echo "  Iniciando Descarga"
echo "========================================================================"
echo ""
echo "Competencia: $COMPETITION"
echo "URL: https://www.kaggle.com/competitions/$COMPETITION/data"
echo "Tamaño esperado: ~170 GB"
echo "Archivo destino: $OUTPUT_FILE"
echo ""
echo "⏱️  Esto tardará 2-6 horas dependiendo de tu conexión"
echo ""

# URL de la API de Kaggle
API_URL="https://www.kaggle.com/api/v1/competitions/data/download-all/$COMPETITION"

START_TIME=$(date +%s)

echo "Descargando con curl..."
echo ""
echo "Comando: curl -L -u \"$KAGGLE_USERNAME:****\" -o \"$OUTPUT_FILE\" \"$API_URL\""
echo ""

# Descargar con curl
# -L: seguir redirects
# -u: autenticación básica con usuario:password
# -o: archivo de salida
# -C -: continuar descarga interrumpida
# --progress-bar: mostrar barra de progreso

if curl -L \
    -u "$KAGGLE_USERNAME:$KAGGLE_KEY" \
    -o "$OUTPUT_FILE" \
    -C - \
    --progress-bar \
    "$API_URL"; then
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    
    echo ""
    echo "✅ Descarga completada en ${HOURS}h ${MINUTES}m"
else
    EXIT_CODE=$?
    echo ""
    echo "❌ ERROR: La descarga falló con código $EXIT_CODE"
    echo ""
    
    if [ -f "$OUTPUT_FILE" ]; then
        PARTIAL_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "Archivo parcial guardado: $PARTIAL_SIZE"
        echo ""
        echo "Puedes reintentar la descarga y se continuará desde donde quedó"
        echo "  bash $0"
    fi
    
    exit $EXIT_CODE
fi

################################################################################
# Verificar descarga
################################################################################

echo ""
echo "Verificando archivo descargado..."

if [ ! -f "$OUTPUT_FILE" ]; then
    echo "❌ ERROR: No se encontró el archivo descargado"
    exit 1
fi

FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
FILE_SIZE_BYTES=$(stat -c%s "$OUTPUT_FILE")

echo "✅ Archivo: $OUTPUT_FILE"
echo "   Tamaño: $FILE_SIZE ($FILE_SIZE_BYTES bytes)"
echo ""

# Verificar que no es un archivo de error HTML
if file "$OUTPUT_FILE" | grep -q "HTML"; then
    echo "❌ ERROR: El archivo descargado parece ser HTML (error de Kaggle)"
    echo ""
    echo "Esto puede ocurrir si:"
    echo "  1. No has aceptado las reglas de la competencia"
    echo "  2. Las credenciales son incorrectas"
    echo "  3. No tienes acceso a la competencia"
    echo ""
    echo "Verifica en: https://www.kaggle.com/competitions/$COMPETITION"
    exit 1
fi

if [ "$FILE_SIZE_BYTES" -lt 1000000 ]; then
    echo "⚠️  Advertencia: El archivo es muy pequeño (< 1 MB)"
    echo "   Esto podría indicar un problema"
    echo ""
    echo "Primeras líneas del archivo:"
    head -20 "$OUTPUT_FILE"
    exit 1
fi

echo "✅ El archivo parece válido (archivo ZIP)"
echo ""

################################################################################
# Preguntar si descomprimir
################################################################################

echo "========================================================================"
echo "  Descompresión"
echo "========================================================================"
echo ""
read -p "¿Descomprimir ahora? (s/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Ss]$ ]]; then
    echo ""
    echo "Descomprimiendo..."
    echo "⏱️  Esto puede tardar 30-60 minutos"
    echo ""
    
    EXTRACT_START=$(date +%s)
    
    if unzip -q "$OUTPUT_FILE"; then
        EXTRACT_END=$(date +%s)
        EXTRACT_ELAPSED=$((EXTRACT_END - EXTRACT_START))
        EXTRACT_MIN=$((EXTRACT_ELAPSED / 60))
        
        echo ""
        echo "✅ Descompresión completada en ${EXTRACT_MIN} minutos"
        echo ""
        
        # Verificar estructura
        if [ -d "ILSVRC" ]; then
            echo "✅ Directorio ILSVRC encontrado"
            echo ""
            
            if [ -d "ILSVRC/Data/CLS-LOC/train" ]; then
                TRAIN_COUNT=$(find ILSVRC/Data/CLS-LOC/train -type f -name "*.JPEG" | wc -l)
                echo "  Train: $TRAIN_COUNT imágenes"
            fi
            
            if [ -d "ILSVRC/Data/CLS-LOC/val" ]; then
                VAL_COUNT=$(find ILSVRC/Data/CLS-LOC/val -type f -name "*.JPEG" | wc -l)
                echo "  Val: $VAL_COUNT imágenes"
            fi
            
            if [ -d "ILSVRC/Data/CLS-LOC/test" ]; then
                TEST_COUNT=$(find ILSVRC/Data/CLS-LOC/test -type f -name "*.JPEG" | wc -l)
                echo "  Test: $TEST_COUNT imágenes"
            fi
            
            echo ""
            echo "Tamaño del dataset:"
            du -sh ILSVRC
            echo ""
            
            # Preguntar si eliminar ZIP
            echo "El archivo ZIP ocupa: $FILE_SIZE"
            read -p "¿Eliminar el archivo ZIP para ahorrar espacio? (s/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Ss]$ ]]; then
                rm "$OUTPUT_FILE"
                echo "✅ Archivo ZIP eliminado"
            fi
        else
            echo "⚠️  No se encontró el directorio ILSVRC"
            echo "Contenido del directorio:"
            ls -lh
        fi
    else
        echo ""
        echo "❌ ERROR: Falló la descompresión"
        echo ""
        echo "Puedes intentar manualmente:"
        echo "  cd $KAGGLE_DIR"
        echo "  unzip $OUTPUT_FILE"
    fi
else
    echo ""
    echo "Descompresión omitida"
    echo ""
    echo "Para descomprimir después:"
    echo "  cd $KAGGLE_DIR"
    echo "  unzip $OUTPUT_FILE"
fi

################################################################################
# Resumen final
################################################################################

TOTAL_END=$(date +%s)
TOTAL_ELAPSED=$((TOTAL_END - START_TIME))
TOTAL_HOURS=$((TOTAL_ELAPSED / 3600))
TOTAL_MIN=$(((TOTAL_ELAPSED % 3600) / 60))

echo ""
echo "========================================================================"
echo "  ✅ DESCARGA COMPLETADA"
echo "========================================================================"
echo ""
echo "Tiempo total: ${TOTAL_HOURS}h ${TOTAL_MIN}m"
echo "Ubicación: $KAGGLE_DIR"
echo ""
echo "Próximos pasos:"
echo ""
echo "1. Verificar dataset:"
echo "   cd $KAGGLE_DIR"
echo "   ls -lh ILSVRC/Data/CLS-LOC/"
echo ""
echo "2. Entrenar VAR:"
echo "   cd $PROJECT_DIR"
echo "   sbatch train_var.slurm"
echo ""
echo "========================================================================"
echo "Fin: $(date)"
echo "========================================================================"

exit 0
