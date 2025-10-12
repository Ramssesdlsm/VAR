#!/usr/bin/env python3
"""
Script de entrenamiento para el modelo VAR (Visual Autoregressive) con ImageNet.
Este script maneja:
- Entrenamiento distribuido con DDP (DistributedDataParallel)
- Carga del VQ-VAE preentrenado
- Optimización del transformer autorregresivo
- Logging con TensorBoard
- Checkpointing y recuperación
"""

import argparse
import glob as glob_module
import os
import sys
import time
from typing import Optional, Tuple

import torch
import torch.distributed as tdist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Importar módulos locales
import dist
from models import VAR, VQVAE
from trainer import VARTrainer
from utils.amp_sc import AmpOptimizer
from utils.data import build_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.lr_control import filter_params, lr_wd_annealing
from utils.misc import MetricLogger, TensorboardLogger


def get_args_parser():
    """Parse argumentos de línea de comandos."""
    parser = argparse.ArgumentParser('VAR training', add_help=False)
    
    # Rutas de datos y modelo
    parser.add_argument('--data_path', default='/path/to/imagenet', type=str,
                        help='Ruta al dataset ImageNet')
    parser.add_argument('--vae_ckpt', default='vae_ch160v4096z32.pth', type=str,
                        help='Checkpoint del VQ-VAE preentrenado')
    parser.add_argument('--exp_name', default='var_imagenet', type=str,
                        help='Nombre del experimento')
    parser.add_argument('--output_dir', default='./output', type=str,
                        help='Directorio para guardar checkpoints y logs')
    
    # Arquitectura del modelo
    parser.add_argument('--depth', default=16, type=int,
                        help='Profundidad del transformer VAR')
    parser.add_argument('--embed_dim', default=1024, type=int,
                        help='Dimensión de embeddings del transformer')
    parser.add_argument('--num_heads', default=16, type=int,
                        help='Número de attention heads')
    parser.add_argument('--mlp_ratio', default=4.0, type=float,
                        help='Ratio de expansión MLP')
    parser.add_argument('--drop_rate', default=0.0, type=float,
                        help='Dropout rate')
    parser.add_argument('--attn_drop_rate', default=0.0, type=float,
                        help='Attention dropout rate')
    parser.add_argument('--drop_path_rate', default=0.0, type=float,
                        help='Stochastic depth rate')
    
    # Parámetros de entrenamiento
    parser.add_argument('--epochs', default=250, type=int,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size por GPU')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations')
    
    # Optimización
    parser.add_argument('--lr', default=1e-4, type=float,
                        help='Learning rate base')
    parser.add_argument('--min_lr', default=0.0, type=float,
                        help='Lower lr bound for cyclic schedulers')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='Epochs to warmup LR')
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help='Weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', default=0.0, type=float,
                        help='Final weight decay')
    parser.add_argument('--grad_clip', default=2.0, type=float,
                        help='Gradient clipping max norm')
    parser.add_argument('--label_smooth', default=0.0, type=float,
                        help='Label smoothing')
    
    # Precisión mixta
    parser.add_argument('--fp16', action='store_true',
                        help='Use fp16 mixed precision')
    parser.add_argument('--bf16', action='store_true',
                        help='Use bf16 mixed precision')
    
    # Dataset
    parser.add_argument('--input_size', default=256, type=int,
                        help='Tamaño de entrada de imagen')
    parser.add_argument('--hflip', action='store_true',
                        help='Usar horizontal flip augmentation')
    parser.add_argument('--mid_reso', default=1.125, type=float,
                        help='Multiplicador para resize antes de crop')
    
    # Entrenamiento distribuido
    parser.add_argument('--world_size', default=1, type=int,
                        help='Número de procesos distribuidos')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='Local rank para entrenamiento distribuido')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='URL para configurar entrenamiento distribuido')
    
    # Logging y checkpoint
    parser.add_argument('--print_freq', default=100, type=int,
                        help='Frecuencia de print')
    parser.add_argument('--save_freq', default=10, type=int,
                        help='Frecuencia de guardado de checkpoints (epochs)')
    parser.add_argument('--resume', default='', type=str,
                        help='Ruta al checkpoint para resumir')
    parser.add_argument('--auto_resume', action='store_true',
                        help='Auto-resume desde el último checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int,
                        help='Época de inicio')
    
    # Evaluación
    parser.add_argument('--eval_freq', default=5, type=int,
                        help='Frecuencia de evaluación (epochs)')
    
    # Otros
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Número de workers para data loading')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory en DataLoader')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed')
    
    return parser


def build_vae(args) -> VQVAE:
    """Cargar el VQ-VAE preentrenado."""
    print(f'[VAE] Cargando VQ-VAE desde {args.vae_ckpt}')
    
    # Verificar que existe el checkpoint
    if not os.path.exists(args.vae_ckpt):
        raise FileNotFoundError(f'VQ-VAE checkpoint no encontrado: {args.vae_ckpt}')
    
    # Crear modelo VQVAE con configuración por defecto
    vae = VQVAE(
        vocab_size=4096,
        z_channels=32,
        ch=160,
        test_mode=True  # Congelar parámetros
    )
    
    # Cargar pesos (weights_only=False es seguro aquí porque controlamos el checkpoint)
    checkpoint = torch.load(args.vae_ckpt, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    vae.load_state_dict(state_dict, strict=True)
    print(f'[VAE] VQ-VAE cargado exitosamente')
    print(f'[VAE] Vocab size: {vae.vocab_size}, patch_nums: {vae.patch_nums}')
    
    return vae


def build_var_model(vae: VQVAE, args) -> VAR:
    """Construir el modelo VAR."""
    print(f'[VAR] Construyendo modelo VAR con depth={args.depth}')
    
    # Calcular max_seq_len basado en patch_nums
    max_seq_len = sum(pn * pn for pn in vae.patch_nums)
    
    # Configuración del transformer
    transformer_config = {
        'depth': args.depth,
        'dim': args.embed_dim,
        'num_heads': args.num_heads,
        'mlp_ratio': args.mlp_ratio,
        'attn_dropout': args.attn_drop_rate,
        'out_dropout': args.drop_rate,
        'ffn_dropout': args.drop_rate,
        'num_classes': 1000,  # ImageNet tiene 1000 clases
        'vocab_size': vae.vocab_size,
        'max_seq_len': max_seq_len,
        'num_levels': len(vae.patch_nums),
        'first_scale_tokens': vae.patch_nums[0] ** 2,
        'Cvae': vae.Cvae,  # Dimensión de embeddings del VQ-VAE
    }
    
    # Crear modelo VAR
    model = VAR(vqvae_model=vae, transformer_config=transformer_config)
    
    print(f'[VAR] Modelo VAR creado exitosamente')
    print(f'[VAR] Parámetros totales: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
    print(f'[VAR] Parámetros entrenables: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M')
    
    return model


def train_one_epoch(
    model: DDP,
    trainer: VARTrainer,
    data_loader: DataLoader,
    optimizer: AmpOptimizer,
    device: torch.device,
    epoch: int,
    args,
    tb_logger: Optional[TensorboardLogger] = None
) -> dict:
    """Entrenar una época."""
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    # Los medidores se crean automáticamente con update()
    header = f'Epoch: [{epoch}]'
    
    accum_iter = args.accum_iter
    
    for data_iter_step, (images, labels) in metric_logger.log_every(
        start_it=0, 
        max_iters=len(data_loader), 
        itrt=iter(data_loader), 
        print_freq=args.print_freq, 
        header=header
    ):
        # Ajustar learning rate y weight decay
        cur_it = len(data_loader) * epoch + data_iter_step
        lr_wd_annealing(
            sche_type='cos',  # Cosine annealing
            optimizer=optimizer.optimizer,
            peak_lr=args.lr,
            wd=args.weight_decay,
            wd_end=args.weight_decay_end,
            cur_it=cur_it,
            wp_it=args.warmup_epochs * len(data_loader),
            max_it=args.epochs * len(data_loader),
        )
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        # Determinar si hacer backward/step en esta iteración
        stepping = (data_iter_step + 1) % accum_iter == 0 or (data_iter_step + 1) == len(data_loader)
        
        # Train step
        grad_norm, scale_log2 = trainer.train_step(
            it=data_iter_step,
            g_it=cur_it,
            stepping=stepping,
            metric_lg=metric_logger,
            tb_lg=tb_logger,
            inp_B3HW=images,
            label_B=labels,
            prog_si=-1,  # No progressive training por ahora
            prog_wp_it=1,  # Evitar división por cero
        )
        
        # Logging
        if stepping:
            lr = optimizer.optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            if grad_norm is not None:
                metric_logger.update(grad_norm=grad_norm)
    
    # Sincronizar métricas
    metric_logger.synchronize_between_processes()
    print(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(trainer: VARTrainer, data_loader: DataLoader, device: torch.device, epoch: int, args):
    """Evaluar el modelo."""
    print(f'[Eval] Evaluando época {epoch}')
    
    L_mean, L_tail, acc_mean, acc_tail, tot, cost = trainer.eval_ep(data_loader)
    
    if dist.is_master():
        print(f'[Eval] Epoch {epoch}: L_mean={L_mean:.4f}, L_tail={L_tail:.4f}, '
              f'Acc_mean={acc_mean:.2f}%, Acc_tail={acc_tail:.2f}%, '
              f'samples={tot}, time={cost:.2f}s')
    
    return {
        'L_mean': L_mean,
        'L_tail': L_tail,
        'acc_mean': acc_mean,
        'acc_tail': acc_tail,
    }


def main(args):
    """Función principal de entrenamiento."""
    # Inicializar entrenamiento distribuido
    dist.initialize(fork=False, timeout=30)
    dist.barrier()
    device = dist.get_device()
    
    # Seed para reproducibilidad
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    print(f'Job dir: {os.getcwd()}')
    print(f'Args: {args}')
    
    # Crear directorio de salida
    if dist.is_master():
        os.makedirs(args.output_dir, exist_ok=True)
        log_dir = os.path.join(args.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
    
    # Construir dataset
    print(f'[Data] Construyendo dataset desde {args.data_path}')
    num_classes, train_dataset, val_dataset = build_dataset(
        data_path=args.data_path,
        final_reso=args.input_size,
        hflip=args.hflip,
        mid_reso=args.mid_reso,
    )
    
    # Verificar número de clases
    assert num_classes == 1000, f'Expected 1000 classes for ImageNet, got {num_classes}'
    
    # Samplers distribuidos
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    train_sampler = DistInfiniteBatchSampler(
        world_size=world_size,
        rank=rank,
        dataset_len=len(train_dataset),
        glb_batch_size=args.batch_size * world_size,
        same_seed_for_all_ranks=0,
        shuffle=True,
        fill_last=False,
        start_ep=0,
        start_it=0,
    )
    val_sampler = EvalDistributedSampler(
        dataset=val_dataset,
        num_replicas=world_size,
        rank=rank,
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    
    # Construir modelos
    vae = build_vae(args).to(device)
    # Forzar VAE a float32 (no mixed precision en modelo congelado)
    vae = vae.float()
    var_model = build_var_model(vae, args).to(device)
    
    # Wrap con DDP
    if dist.initialized():
        var_ddp = DDP(
            var_model,
            device_ids=[dist.get_local_rank()],
            find_unused_parameters=False,
        )
    else:
        var_ddp = var_model
    
    # Optimizador
    names, paras, param_groups_raw = filter_params(
        var_ddp,
        nowd_keys={'cls_token', 'start_token', 'task_token', 'cfg_uncond',
                   'pos_embed', 'pos_1LC', 'pos_start', 'gamma', 'beta',
                   'ada_gss', 'moe_bias', 'scale_mul'}
    )
    
    # Limpiar param_groups para AdamW (remover campos personalizados)
    param_groups = []
    for pg in param_groups_raw:
        # Extraer weight_decay basado en wd_sc
        wd = args.weight_decay * pg.get('wd_sc', 1.0)
        param_groups.append({
            'params': pg['params'],
            'weight_decay': wd,
            'lr': args.lr * pg.get('lr_sc', 1.0)
        })
    
    # Determinar mixed_precision como entero
    if args.fp16:
        mixed_precision = 1  # fp16
    elif args.bf16:
        mixed_precision = 2  # bf16
    else:
        mixed_precision = 0  # none
    
    optimizer = AmpOptimizer(
        mixed_precision=mixed_precision,
        optimizer=torch.optim.AdamW(param_groups, betas=(0.9, 0.95)),
        names=names,
        paras=paras,
        grad_clip=args.grad_clip,
        n_gradient_accumulation=args.accum_iter,
    )
    
    # Crear trainer
    trainer = VARTrainer(
        device=device,
        patch_nums=vae.patch_nums,
        resos=tuple(pn * vae.downsample for pn in vae.patch_nums),
        vae_local=vae,
        var_wo_ddp=var_model,
        var=var_ddp,
        var_opt=optimizer,
        label_smooth=args.label_smooth,
    )
    
    # TensorBoard logger
    tb_logger = None
    if dist.is_master():
        tb_logger = TensorboardLogger(
            log_dir=os.path.join(args.output_dir, 'logs'),
            filename_suffix=f'_d{args.depth}'
        )
    
    # Auto-resume o cargar checkpoint
    start_epoch = args.start_epoch
    if args.auto_resume:
        # Buscar el último checkpoint
        ckpts = sorted(glob_module.glob(os.path.join(args.output_dir, 'checkpoint-*.pth')), 
                      key=os.path.getmtime, reverse=True)
        if ckpts:
            args.resume = ckpts[0]
            print(f'[Auto-resume] Encontrado checkpoint: {args.resume}')
    
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location='cpu')
        trainer.load_state_dict(checkpoint['trainer'], strict=True)
        start_epoch = checkpoint['epoch'] + 1
        print(f'[Resume] Resumiendo desde época {start_epoch}')
    
    # Loop de entrenamiento
    print(f'[Train] Iniciando entrenamiento desde época {start_epoch}')
    start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        if hasattr(train_loader.batch_sampler, 'set_epoch'):
            train_loader.batch_sampler.set_epoch(epoch)
        
        # Entrenar una época
        train_stats = train_one_epoch(
            model=var_ddp,
            trainer=trainer,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            args=args,
            tb_logger=tb_logger,
        )
        
        # Evaluar
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            eval_stats = evaluate(trainer, val_loader, device, epoch, args)
            if tb_logger is not None:
                tb_logger.update(head='eval', **eval_stats, step=epoch)
        
        # Guardar checkpoint
        if dist.is_master() and ((epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1):
            checkpoint = {
                'epoch': epoch,
                'trainer': trainer.state_dict(),
                'args': args,
            }
            checkpoint_path = os.path.join(args.output_dir, f'checkpoint-{epoch:04d}.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'[Save] Checkpoint guardado en {checkpoint_path}')
            
            # Guardar también el último
            last_checkpoint_path = os.path.join(args.output_dir, 'checkpoint-last.pth')
            torch.save(checkpoint, last_checkpoint_path)
    
    total_time = time.time() - start_time
    print(f'[Train] Entrenamiento completado en {total_time / 3600:.2f} horas')
    
    if tb_logger is not None:
        tb_logger.flush()
        tb_logger.close()


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Validar argumentos
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
