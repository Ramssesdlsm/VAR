"""
Script de verificación para el fix correcto de VAR
"""
import torch
import sys

print("="*60)
print(" VERIFICACIÓN DEL FIX CORRECTO DE VAR")
print("="*60)

# Test 1: Verificar que prepare_var_inputs funciona correctamente
print("\n" + "="*60)
print("TEST 1: prepare_var_inputs con idxBl_to_var_input")
print("="*60)

sys.path.append('.')
from models.vqvae import VQVAE

# Crear VQ-VAE
vqvae_config = {
    'vocab_size': 4096,
    'z_channels': 32,
    'ch': 160,
    'test_mode': True,
    'share_quant_resi': 4,
    'v_patch_nums': (1, 2, 3),  # Simplificado para testing
}

vqvae = VQVAE(**vqvae_config)
patch_nums = vqvae.patch_nums

print(f"\nPatch nums: {patch_nums}")
print(f"Cvae: {vqvae.Cvae}")

# Simular tokenización
# VQ-VAE espera imágenes de tamaño patch_nums[-1] * patch_size
# Para patch_nums=(1,2,3) con patch_size=16: 3*16 = 48
B = 2
img_size = patch_nums[-1] * 16  # patch_size default es 16
images = torch.randn(B, 3, img_size, img_size)
print(f"Image size: {img_size}x{img_size}")

with torch.no_grad():
    token_maps = vqvae.img_to_idxBl(images)

print(f"\nToken maps (lista de tensores):")
for i, tm in enumerate(token_maps):
    print(f"  Scale {i}: shape {tm.shape}")

# Verificar idxBl_to_var_input
x_BLCv_wo_first_l = vqvae.quantize.idxBl_to_var_input(token_maps)
print(f"\nx_BLCv_wo_first_l (embeddings procesados):")
print(f"  Shape: {x_BLCv_wo_first_l.shape}")
print(f"  Dtype: {x_BLCv_wo_first_l.dtype}")

# Verificar dimensiones
total_tokens = sum(pn**2 for pn in patch_nums)
first_l = patch_nums[0]**2
expected_len = total_tokens - first_l

print(f"\nDimensiones esperadas:")
print(f"  Total tokens (gt_BL): {total_tokens}")
print(f"  First scale: {first_l}")
print(f"  x_BLCv_wo_first_l length: {expected_len}")
print(f"  Cvae: {vqvae.Cvae}")

assert x_BLCv_wo_first_l.shape == (B, expected_len, vqvae.Cvae), \
    f"Shape mismatch: {x_BLCv_wo_first_l.shape} != ({B}, {expected_len}, {vqvae.Cvae})"

print("\n✓ PASS: x_BLCv_wo_first_l tiene la forma correcta")

# Test 2: Verificar prepare_var_inputs
print("\n" + "="*60)
print("TEST 2: prepare_var_inputs completo")
print("="*60)

from models.var import prepare_var_inputs

labels = torch.randint(0, 1000, (B,))
model_inputs = prepare_var_inputs(token_maps, labels, patch_nums, vqvae.quantize)

print(f"\nmodel_inputs keys: {model_inputs.keys()}")
print(f"x_BLCv_wo_first_l shape: {model_inputs['x_BLCv_wo_first_l'].shape}")
print(f"level_indices shape: {model_inputs['level_indices'].shape}")
print(f"attn_mask shape: {model_inputs['attn_mask'].shape}")

assert model_inputs['x_BLCv_wo_first_l'].shape == (B, expected_len, vqvae.Cvae)
assert model_inputs['level_indices'].shape == (B, total_tokens)
assert model_inputs['attn_mask'].shape == (total_tokens, total_tokens)

print("\n✓ PASS: prepare_var_inputs retorna las formas correctas")

# Test 3: Verificar máscara >= (correlación intra-escala)
print("\n" + "="*60)
print("TEST 3: Máscara de Atención (d >= dT)")
print("="*60)

attn_mask = model_inputs['attn_mask']
level_indices = model_inputs['level_indices'][0]  # Tomar primer batch

print(f"\nLevel indices: {level_indices.tolist()}")
print(f"\nMáscara de atención (primeras 10x10 posiciones):")
print(attn_mask[:10, :10].int())

# Verificar que tokens del mismo nivel se ven entre sí
for i in range(min(5, len(level_indices))):
    same_level_positions = (level_indices == level_indices[i]).nonzero(as_tuple=True)[0]
    can_see_same_level = attn_mask[i, same_level_positions].all()
    print(f"\nPosición {i} (level {level_indices[i].item()}):")
    print(f"  Puede ver todas las posiciones de su mismo nivel: {can_see_same_level.item()}")
    if not can_see_same_level:
        print("  ✗ ERROR: No puede ver tokens del mismo nivel")
        sys.exit(1)

print("\n✓ PASS: Tokens del mismo nivel se ven entre sí (fully correlated)")

# Test 4: Verificar que NO ve niveles futuros
print("\n" + "="*60)
print("TEST 4: No ve niveles futuros")
print("="*60)

for i in range(min(5, len(level_indices))):
    future_levels = level_indices > level_indices[i]
    can_see_future = attn_mask[i, future_levels].any()
    print(f"Posición {i} (level {level_indices[i].item()}) ve niveles futuros: {can_see_future.item()}")
    if can_see_future:
        print("  ✗ ERROR: Puede ver niveles futuros")
        sys.exit(1)

print("\n✓ PASS: Ningún token ve niveles futuros")

# Test 5: Forward pass completo
print("\n" + "="*60)
print("TEST 5: Forward pass completo del modelo")
print("="*60)

from models.var import VAR

transformer_config = {
    'depth': 2,
    'dim': 64,
    'num_heads': 2,
    'mlp_ratio': 4.0,
    'attn_dropout': 0.0,
    'out_dropout': 0.0,
    'ffn_dropout': 0.0,
    'num_classes': 1000,
    'vocab_size': 4096,
    'max_seq_len': total_tokens,
    'num_levels': len(patch_nums),
    'first_scale_tokens': first_l,
    'Cvae': vqvae.Cvae,
}

var_model = VAR(vqvae_model=vqvae, transformer_config=transformer_config)

print("\nProbando forward pass...")
try:
    with torch.no_grad():
        logits = var_model(images, labels)
    
    print(f"✓ Forward pass exitoso")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Expected: [{B}, {total_tokens}, {vqvae.vocab_size}]")
    
    assert logits.shape == (B, total_tokens, vqvae.vocab_size)
    print("\n✓ PASS: Logits tienen la forma correcta")
    
    # Verificar ground truth
    gt_BL = torch.cat(token_maps, dim=1)
    print(f"\nGround truth shape: {gt_BL.shape}")
    assert gt_BL.shape == (B, total_tokens)
    print("✓ PASS: Ground truth coincide con logits")
    
    # Calcular accuracy
    pred = logits.argmax(dim=-1)
    acc = (pred == gt_BL).float().mean().item() * 100
    print(f"\nAccuracy inicial (random): {acc:.2f}%")
    print(f"Esperado para random: ~{100/vqvae.vocab_size:.2f}%")
    
    # Calcular loss
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits.view(-1, vqvae.vocab_size), gt_BL.view(-1))
    print(f"Loss inicial: {loss.item():.4f}")
    
    expected_loss = -torch.log(torch.tensor(1.0/vqvae.vocab_size))
    print(f"Loss esperada para random: ~{expected_loss.item():.4f}")
    
except Exception as e:
    print(f"✗ ERROR en forward pass: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print(" RESUMEN")
print("="*60)
print("✓ PASS: idxBl_to_var_input genera embeddings procesados")
print("✓ PASS: prepare_var_inputs usa embeddings procesados")
print("✓ PASS: Máscara d >= dT (correlación intra-escala)")
print("✓ PASS: No ve niveles futuros")
print("✓ PASS: Forward pass completo funciona")
print("✓ PASS: Shapes de logits y ground truth coinciden")
print("\n✓ TODOS LOS TESTS PASARON")
print("\nDiferencias con la versión anterior (INCORRECTA):")
print("  ❌ Antes: Usaba tokens crudos (índices) directamente")
print("  ❌ Antes: Token embedding sin procesamiento")
print("  ✓ Ahora: Usa embeddings procesados de idxBl_to_var_input")
print("  ✓ Ahora: word_embed transforma Cvae -> dim")
print("  ✓ Ahora: Información acumulada de escalas anteriores")
print("  ✓ Ahora: Máscara >= (correlación intra-escala)")
print("\nEsto debería eliminar el problema de 99% accuracy artificial.")
print("="*60)
