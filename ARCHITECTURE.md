# Detailed Model Architecture - Fusion ResNet50-UNet

## Overview

The `Resnet50_UNet` model in `model.py` is a **Dual-Branch Encoder-Decoder** architecture specifically designed for **flood detection** from multi-modal data (SAR + DEM/JRC).

## Overall Architecture Diagram

```
┌─────────────────┐         ┌─────────────────┐
│   Input 1       │         │   Input 2       │
│  (512×512×3)    │         │  (512×512×3)    │
│  SAR/Optical    │         │   DEM/JRC       │
└─────────┬───────┘         └─────────┬───────┘
          │                           │
          ▼                           ▼
┌─────────────────┐         ┌─────────────────┐
│ ResNet50        │         │ ResNet50        │
│ Encoder 1       │         │ Encoder 2       │
│ (Pretrained)    │         │ (Pretrained)    │
└─────────┬───────┘         └─────────┬───────┘
          │                           │
          │    ┌───────────────┐      │
          └────┤ Element-wise  ├──────┘
               │     Add       │
               └───────┬───────┘
                       │
                       ▼
               ┌───────────────┐
               │   U-Net       │
               │   Decoder     │
               │ (Skip Conn)   │
               └───────┬───────┘
                       │
                       ▼
               ┌───────────────┐
               │   Output      │
               │ (256×256×1)   │
               │ Flood Mask    │
               └───────────────┘
```

## Component Details

### 1. Input Processing

**Input 1**: 
- Shape: `(batch, 512, 512, 3)`
- Data Type: SAR imagery from Sentinel-1 or optical imagery
- Preprocessing: Normalization, scaling

**Input 2**: 
- Shape: `(batch, 512, 512, 3)` 
- Data Type: DEM (Digital Elevation Model) and JRC water data
- Preprocessing: Height normalization, water probability encoding

### 2. ResNet50 Encoder Architecture

Each encoder follows standard ResNet50 structure:

#### Stage 1: Initial Convolution
```
Input (512×512×3)
    ↓ ZeroPad(3×3) + Conv(7×7, stride=2) 
    ↓ BatchNorm + ReLU + MaxPool(3×3, stride=2)
    → f1: (128×128×64)
```

#### Stage 2: Residual Blocks  
```
f1 (128×128×64)
    ↓ conv_block([64,64,256], stride=1)
    ↓ identity_block([64,64,256]) × 2  
    → f2: (64×64×256) [after one_side_pad]
```

#### Stage 3: Residual Blocks
```
f2 (64×64×256)  
    ↓ conv_block([128,128,512], stride=2)
    ↓ identity_block([128,128,512]) × 3
    → f3: (32×32×512)
```

#### Stage 4: Residual Blocks  
```
f3 (32×32×512)
    ↓ conv_block([256,256,1024], stride=2)  
    ↓ identity_block([256,256,1024]) × 5
    → f4: (16×16×1024)
```

### 3. Feature Fusion Strategy

After obtaining feature maps from both encoders, fusion is performed using **element-wise addition**:

```python
# Feature fusion at each scale
f1_fused = Add()([f11, f21])  # (128×128×64)
f2_fused = Add()([f12, f22])  # (64×64×256)  
f3_fused = Add()([f13, f23])  # (32×32×512)
f4_fused = Add()([f14, f24])  # (16×16×1024)
```

**Why Addition instead of Concatenation:**
- Reduces number of parameters
- Prevents overfitting  
- Encourages learning complementary features
- Better memory and computational efficiency

### 4. U-Net Decoder with Skip Connections

#### Decoder Layer 1
```
f4_fused (16×16×1024)
    ↓ ZeroPad(1×1) + Conv(3×3, 512) + BN + ReLU
    ↓ UpSample2D(2×2) → (32×32×512)
    ↓ Concatenate with f3_fused → (32×32×1024)
```

#### Decoder Layer 2  
```
Concat output (32×32×1024)
    ↓ ZeroPad(1×1) + Conv(3×3, 256) + BN + ReLU
    ↓ UpSample2D(2×2) → (64×64×256)
    ↓ Concatenate with f2_fused → (64×64×512)
```

#### Decoder Layer 3
```  
Concat output (64×64×512)
    ↓ ZeroPad(1×1) + Conv(3×3, 128) + BN + ReLU
    ↓ UpSample2D(2×2) → (128×128×128)
    ↓ Concatenate with f1_fused (if l1_skip_conn=True) → (128×128×192)
```

#### Final Decoder Layer
```
Skip connection output (128×128×192)
    ↓ ZeroPad(1×1) + Conv(3×3, 64) + BN + ReLU  
    ↓ UpSample2D(2×2) → (256×256×64)
    ↓ Conv(1×1, n_classes) + Sigmoid → (256×256×1)
```

### 5. Detailed Residual Blocks

#### Identity Block
```python
def identity_block(input, filters):
    f1, f2, f3 = filters
    
    # Branch 2a: 1×1 conv 
    x = Conv2D(f1, (1,1))(input)
    x = BatchNorm()(x) 
    x = ReLU()(x)
    
    # Branch 2b: 3×3 conv
    x = Conv2D(f2, (3,3), padding='same')(x)
    x = BatchNorm()(x)
    x = ReLU()(x)
    
    # Branch 2c: 1×1 conv  
    x = Conv2D(f3, (1,1))(x)
    x = BatchNorm()(x)
    
    # Shortcut connection
    x = Add()([x, input])
    x = ReLU()(x)
    return x
```

#### Convolution Block  
```python
def conv_block(input, filters, stride=2):
    f1, f2, f3 = filters
    
    # Main path (similar to identity_block)
    x = Conv2D(f1, (1,1), strides=stride)(input)
    # ... (same as identity_block)
    
    # Shortcut path with projection
    shortcut = Conv2D(f3, (1,1), strides=stride)(input)
    shortcut = BatchNorm()(shortcut)
    
    x = Add()([x, shortcut])  
    x = ReLU()(x)
    return x
```

### 6. Pretrained Weights Loading

```python
if pretrained == 'imagenet':
    weights_path = keras.utils.get_file(
        pretrained_url.split("/")[-1], 
        pretrained_url
    )
    Model(img_input, x).load_weights(
        weights_path, 
        by_name=True, 
        skip_mismatch=True
    )
```

**Why `skip_mismatch=True`?**
- Input channels might differ from ImageNet (3 channels)
- Some layers might be modified 
- Only load weights for matching layers

### 7. Training Strategy

#### Phase 1: Frozen Encoder Training
```python
# Freeze all layers except decoder
for layer in model.layers:
    if 'DEC_' not in layer.name:
        layer.trainable = False

# Train 2 epochs with high learning rate
model.fit(..., epochs=2)
```

#### Phase 2: Fine-tuning  
```python
# Unfreeze all layers
for layer in model.layers:
    layer.trainable = True
    
# Train 100 epochs with lower learning rate + scheduler
model.fit(..., epochs=100, callbacks=[scheduler])
```

### 8. Loss Function and Metrics

```python
# Combination loss
dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = 0.2 * dice_loss + 0.8 * focal_loss

# Metrics
metrics = [
    sm.metrics.IOUScore(threshold=0.5),
    sm.metrics.FScore(threshold=0.5)
]
```

**Rationale for combination loss:**
- **Dice Loss**: Optimized for class imbalance (flood vs non-flood)
- **Focal Loss**: Focus on hard examples (boundary pixels)
- **Weighted combination**: Balance between two objectives

### 9. Model Parameters

| Component | Parameters | Memory (GB) |
|-----------|------------|-------------|
| Encoder 1 | ~23M | ~2.5 |  
| Encoder 2 | ~23M | ~2.5 |
| Decoder | ~2M | ~1.0 |
| **Total** | **~48M** | **~6.0** |

### 10. Inference Pipeline

```python
# Load model and weights  
model = Resnet50_UNet(1, input1_shape, input2_shape)
model.load_weights('best_checkpoint.h5')

# Preprocessing
img1 = scale_img(sar_image)      # SAR normalization
img2 = scale_img(dem_jrc_image)  # DEM/JRC normalization

# Prediction
pred = model.predict([img1, img2])
flood_mask = (pred > 0.5).astype(np.uint8)

# Post-processing (morphology, CRF, etc.)
final_mask = apply_morphology(flood_mask)
```

### 11. Advantages and Limitations

#### Advantages:
- **Multi-modal fusion**: Effective for SAR + topographic data
- **Pretrained benefits**: Transfer learning from ImageNet
- **Multi-scale**: Skip connections preserve spatial details  
- **Flexible**: Can disable skip connections
- **Robust**: Combination loss handles class imbalance

#### Limitations:
- **Memory intensive**: 2 ResNet50 encoders
- **Resolution loss**: Output 256×256 vs input 512×512
- **Fixed fusion**: Element-wise add may not be optimal for all cases
- **Architecture complexity**: Many hyperparameters to tune

### 12. Possible Improvements

1. **Attention mechanisms**: Replace element-wise add with attention fusion
2. **Progressive upsampling**: Add layers to reach full resolution 
3. **Lightweight backbones**: EfficientNet instead of ResNet50
4. **Feature Pyramid**: Multi-scale feature fusion
5. **Self-supervised pretraining**: Pretrain on SAR data instead of ImageNet

---

*This architecture is specifically optimized for flood detection from Sentinel-1 SAR data combined with DEM and JRC water information.*