# Vehicle Classification

Klasifikasi gambar kendaraan menggunakan MobileNetV2 (Transfer Learning) dengan akurasi target ≥ 85%.

## Dataset

- **Sumber:** [aryadytm/vehicle-classification](https://huggingface.co/datasets/aryadytm/vehicle-classification)
- **Jumlah:** 2,933 gambar
- **Kelas:** 17 kategori (Ambulance, Barge, Bicycle, Boat, Bus, Car, Cart, Caterpillar, Helicopter, Limousine, Motorcycle, Segway, Snowmobile, Tank, Taxi, Truck, Van)

## Model

- **Base:** MobileNetV2 (pre-trained ImageNet)
- **Head:** GlobalAveragePooling2D → Dense(256) → Dropout(0.5) → Dense(128) → Dropout(0.3) → Dense(17, softmax)
- **Augmentasi:** MixUp (35% probabilitas per batch)

## Setup

```bash
# Clone repo
git clone https://github.com/rezzz59/Vecihle_classification.git
cd Vecihle_classification

# Download dataset
huggingface-cli download aryadytm/vehicle-classification --repo-type dataset --local-dir ./datasets

# Install dependencies
pip install tensorflow==2.19.0 jupyter numpy matplotlib scikit-learn
```

## Training

1. Buka `vehicle_classification.ipynb` di Jupyter
2. Jalankan semua cell dari atas ke bawah
3. Model disimpan ke `best_model.keras`

**Phase 1:** Frozen base (20 epochs)
**Phase 2:** Fine-tune last 30 layers (25 epochs)

## Output

| File | Keterangan |
|------|-----------|
| `best_model.keras` | Model terbaik |
| `saved_model/` | SavedModel format |
| `tflite/model.tflite` | TFLite model |
| `tfjs_model/` | TensorFlow.js model |
| `accuracy_loss_plot.png` | Plot training |

## Kriteria Submission

- Dataset ≥ 1,000 gambar ✅
- Conv2D + MaxPooling2D ✅
- Train Accuracy ≥ 85%
- Plot accuracy & loss
- Export: SavedModel, TFLite, TFJS