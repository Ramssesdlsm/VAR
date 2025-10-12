# SOLUCIÓN: Problema de 99% Accuracy en VAR

## Fecha
2025-01-XX

## Resumen del Problema

Durante el entrenamiento del modelo VAR, se observó:
- **Loss**: 3.1-3.2 (valor razonable)
- **Accuracy**: 99% desde la época 1-2 (valor imposible)

Este comportamiento se reproducía tanto con el dataset pequeño (200K imágenes) como con el dataset completo de ImageNet, por lo que NO era un problema de overfitting por falta de datos.

## Diagnóstico

### Inconsistencia Matemática
La relación entre loss y accuracy en clasificación multiclase es:
```
Loss = -log(probabilidad_correcta)
Accuracy = probabilidad_correcta * 100
```

Por lo tanto:
```
Loss = 3.1 → probabilidad ≈ exp(-3.1) ≈ 0.045 → Accuracy ≈ 4.5%
```

Pero observábamos **Accuracy = 99%**, lo cual es matemáticamente inconsistente con **Loss = 3.1**.

### Causa Raíz: Bug en la Máscara de Atención

El problema estaba en el archivo `models/var.py`, línea 32:

**CÓDIGO INCORRECTO:**
```python
attn_mask = d.unsqueeze(1) >= d.unsqueeze(0)
```

Esta máscara permitía que tokens del mismo nivel se vieran entre sí durante la atención, causando **data leakage**:

- El modelo recibía `teacher_forcing_tokens = [scale1, scale2, ..., scaleN]` como input
- En la posición i del level L, el modelo podía ver su propio token de input
- Como el token de input ERA el ground truth, el modelo solo tenía que "copiarlo"
- Resultado: 99% accuracy artificial

#### Ejemplo Concreto:
Con `patch_nums = (1, 2, 3)`:
```
Ground truth: [scale0_token, scale1_token0, scale1_token1, ...]
Teacher forcing: [scale1_token0, scale1_token1, ...]
```

Con la máscara `>=`:
```
Posición 1 (predice scale1_token0):
  - Input en posición 1: teacher_forcing[0] = scale1_token0
  - Puede ver: posiciones [0, 1, 2, 3, 4] (todos los tokens de level 0 y 1)
  - ¡Puede ver posición 1 (SÍ MISMO)!
  - Prediction: "veo scale1_token0 en el input, copio ese valor"
  - Accuracy: 100% (pero sin aprender nada real)
```

## Solución Implementada

### Cambio 1: Máscara de Atención Causal Estricta

**Archivo**: `models/var.py`, línea 32

**ANTES (INCORRECTO):**
```python
attn_mask = d.unsqueeze(1) >= d.unsqueeze(0)
```

**DESPUÉS (CORRECTO):**
```python
# Máscara causal: cada token solo puede atender a tokens de niveles ANTERIORES
# d.unsqueeze(1) > d.unsqueeze(0): nivel actual > nivel del token a atender
# Esto asegura que tokens del mismo nivel NO se vean entre sí
attn_mask = d.unsqueeze(1) > d.unsqueeze(0)
```

**Efecto:**
```
Posición 1 (level 1, predice scale1_token0):
  - Input: teacher_forcing[0] = scale1_token0
  - Puede ver: solo posición 0 (scale0)
  - NO puede ver: posiciones 1-4 (scale1 tokens)
  - El modelo debe predecir basándose solo en scale0, no en scale1
```

### Cambio 2: Position Embeddings Consistentes

**Archivo**: `models/transformers.py`, línea 208

**ANTES:**
```python
sos_sequence = sos_tokens + self.pos_start + self.level_embedding(sos_level_indices)
```

**DESPUÉS:**
```python
# Usar position_embedding consistente para todas las posiciones
sos_sequence = sos_tokens + self.position_embedding[:, :self.first_scale_tokens, :] + self.level_embedding(sos_level_indices)
```

**Efecto:**
- Position embeddings ahora son consistentes en toda la secuencia
- Evita discrepancias entre `pos_start` y `position_embedding`

## Verificación

Ejecutar `python test_fix.py` para verificar:

1. **Máscara de Atención**: Tokens del mismo nivel NO se ven entre sí ✓
2. **Alineación**: Teacher forcing y ground truth correctamente alineados ✓
3. **Loss-Accuracy**: Relación matemática correcta esperada ✓

## Resultado Esperado Después del Fix

### Antes del Fix (INCORRECTO):
```
Epoch 1:
  Loss: 3.1
  Accuracy: 99%  ← IMPOSIBLE, indica leakage
```

### Después del Fix (CORRECTO):
```
Epoch 1:
  Loss: 3.1
  Accuracy: 4-5%  ← Consistente con Loss, modelo aprendiendo desde cero

Epoch 10:
  Loss: 2.5
  Accuracy: 15-20%  ← Mejorando gradualmente

Epoch 50:
  Loss: 1.2
  Accuracy: 50-60%  ← Aprendiendo representaciones

Epoch 200:
  Loss: 0.5
  Accuracy: 80%  ← Convergencia esperada según paper
```

## Archivos Modificados

1. **models/var.py**:
   - Línea 32: Cambio de `>=` a `>` en máscara de atención
   - Agregada documentación detallada en `prepare_var_inputs()`

2. **models/transformers.py**:
   - Línea 208: Uso de `position_embedding` consistente
   - Agregada documentación detallada en `forward()`

3. **test_fix.py** (nuevo):
   - Script de verificación de la corrección

## Referencias

- Issue #167 en FoundationVision/VAR: Usuarios reportando ~5% accuracy (esperado inicialmente)
- Paper VAR: Reporta ~82.6% accuracy final (después de entrenar completamente)
- Nuestro problema: 99% accuracy desde época 1 → data leakage confirmado

## Próximos Pasos

1. ✓ Corrección implementada
2. ✓ Verificación con tests
3. ⏳ Ejecutar entrenamiento con código corregido
4. ⏳ Validar que accuracy inicial sea ~4-5%
5. ⏳ Monitorear progreso de entrenamiento
6. ⏳ Confirmar convergencia a ~80% accuracy

**NOTA**: NO enviar jobs con sbatch hasta confirmar que el clúster tenga capacidad disponible.
