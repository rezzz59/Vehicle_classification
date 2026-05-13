# Vehicle Classification - Improved Augmentation Design

**Date:** 2026-05-13
**Author:** Claude
**Status:** Approved

## Goal
Increase model accuracy from 79.1% to 85%+ using MixUp/CutMix augmentation.

## Current Status
- Dataset: 2,933 images, 17 classes
- Current Train Acc: 79.1%
- Target: >= 85%

## Design

### 1. Architecture
- Base: MobileNetV2 (pre-trained, frozen first, then fine-tune last 30 layers)
- Head: GlobalAveragePooling2D → BatchNorm → Dense(256) → Dropout(0.5) → BatchNorm → Dense(128) → Dropout(0.3) → Dense(17, softmax)
- Image size: 224x224

### 2. Augmentation Strategy
MixUp and CutMix applied probabilistically (50% chance per batch):
- **MixUp**: `lambda = uniform(0.2, 0.4)` - blend images and labels
- **CutMix**: Random crop from another image in same batch

Plus standard augmentation:
- Rescale 1./255
- Random rotation 30°
- Width/height shift 0.2
- Shear 0.2
- Zoom 0.3
- Horizontal flip
- Brightness range 0.2
- Channel shift range 20

### 3. Training Configuration

**Phase 1 (Frozen base):**
- Epochs: 20 (increased from 15)
- Optimizer: Adam(lr=0.0005) - reduced from 0.001
- Batch size: 16 (reduced from 32 for better MixUp)
- Callbacks: EarlyStopping(patience=6), ModelCheckpoint, ReduceLROnPlateau(factor=0.5, patience=3)

**Phase 2 (Fine-tune last 30 layers):**
- Epochs: 25 (increased from 20)
- Optimizer: Adam(lr=1e-4)
- Callbacks: Same as Phase 1

### 4. Custom Training Loop
Switch from `model.fit()` to custom training loop to properly implement MixUp/CutMix:
```python
for epoch in range(epochs):
    for batch in dataset:
        if random.random() < 0.5:
            images, labels = mixup(images, labels)
        else if random.random() < 0.5:
            images, labels = cutmix(images, labels)
        # training step
```

### 5. Expected Improvements
- MixUp helps with imbalanced classes (Cart: 51, Limousine: 74)
- Better generalization → higher accuracy
- More stable training with lower LR

### 6. Estimated Time
- Phase 1: ~20 minutes
- Phase 2: ~25 minutes
- Total: ~45 minutes

## Files to Modify
- `vehicle_classification.ipynb` - cells 4 (data gen), 5 (model), 6 (training)

## Output
- Retrained `best_model.keras`
- Updated `training_history.csv`
- Updated `accuracy_loss_plot.png`