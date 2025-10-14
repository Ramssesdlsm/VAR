#!/bin/bash

#SBATCH --partition=GPU
#SBATCH --job-name=VAR_TRAINING
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
##SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=0
#SBATCH --output=/home/est_posgrado_ramsses.delossantos/VAR/logs/train_%j.out
#SBATCH --error=/home/est_posgrado_ramsses.delossantos/VAR/logs/train_%j.err

##########  Variables de entorno ############

# Crear directorio de logs
mkdir -p /home/est_posgrado_ramsses.delossantos/VAR/logs

# Activar conda
CONDA_PATH=$(which conda)
CONDA_BASE=$(dirname $(dirname $CONDA_PATH))
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate nn-pytorch-torchdeq

cd /home/est_posgrado_ramsses.delossantos/VAR

##########  Configuración Multi-Nodo ############

NUM_GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
TOTAL_GPUS=$((SLURM_NNODES * NUM_GPUS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=$(shuf -i 10000-65000 -n 1)
export WORLD_SIZE=$TOTAL_GPUS
export OMP_NUM_THREADS=8

# Configuración para optimizar NCCL
export NCCL_DEBUG=WARN  # Cambiar a INFO para debugging
export NCCL_IB_DISABLE=1  # Deshabilitar InfiniBand si no está disponible
export NCCL_SOCKET_IFNAME=^docker0,lo  # Evitar interfaces de red irrelevantes

echo "================================"
echo "  Configuración Multi-Nodo"
echo "================================"
echo "Job ID: $SLURM_JOBID"
echo "Nodos: $SLURM_NNODES"
echo "Lista: $SLURM_NODELIST"
echo "GPUs/nodo: $NUM_GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "================================"
echo ""

##########  Iniciar Monitor de GPU en Background ############

# Función para monitorear GPU cada 30 segundos
monitor_gpu() {
    GPU_LOG="logs/gpu_monitor_${SLURM_JOBID}.log"
    echo "=== Monitor de GPU iniciado en $(hostname) ===" > "$GPU_LOG"
    
    while true; do
        TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
        echo "=== $TIMESTAMP - $(hostname) ===" >> "$GPU_LOG"
        
        nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu \
            --format=csv,noheader | \
            awk -F', ' '{printf "  GPU %s: Util=%s MemUtil=%s Mem=%s/%s Temp=%s\n", $1, $2, $3, $4, $5, $6}' >> "$GPU_LOG"
        
        sleep 30
    done
}

# Lanzar monitor en background solo en nodo maestro
if [[ $(hostname) == "$MASTER_ADDR" ]]; then
    monitor_gpu &
    MONITOR_PID=$!
    echo "Monitor GPU iniciado con PID: $MONITOR_PID"
fi

##########  Ejecutar Entrenamiento Multi-Nodo ############

srun torchrun \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$NUM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOBID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    train.py \
     --data_path data/imagenet_200train_50val \
    --vae_ckpt vae_ch160v4096z32.pth \
    --exp_name var_d16 \
    --output_dir ./output/var_d16 \
    --depth 16 \
    --embed_dim 1024 \
    --num_heads 16 \
    --epochs 100 \
    --batch_size 64 \
    --accum_iter 1 \
    --lr 2e-4 \
    --weight_decay 0.05 \
    --warmup_epochs 8 \
    --min_lr 5e-7 \
    --grad_clip 3.0 \
    --label_smooth 0.1 \
    --input_size 256 \
    --hflip \
    --fp16 \
    --num_workers 10 \
    --pin_mem \
    --print_freq 100 \
    --save_freq 5 \
    --eval_freq 5 \
    --auto_resume

# Matar monitor al terminar
if [[ ! -z "$MONITOR_PID" ]]; then
    kill $MONITOR_PID 2>/dev/null
fi

echo "Entrenamiento completado"
