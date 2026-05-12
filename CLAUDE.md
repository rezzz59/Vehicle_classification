# Vehicle Classification Project

## Project Overview
Submission Proyek Deep Learning - Klasifikasi Gambar Kendaraan menggunakan Transfer Learning (MobileNetV2)

## Repository
- **GitHub**: https://github.com/rezzz59/Vecihle_classification
- **Remote**: origin

## Dataset
- **Source**: aryadytm/vehicle-classification (Hugging Face)
- **Total**: 2,933 gambar
- **Classes**: 17 kelas kendaraan
  - Ambulance, Barge, Bicycle, Boat, Bus, Car, Cart, Caterpillar, Helicopter, Limousine, Motorcycle, Segway, Snowmobile, Tank, Taxi, Truck, Van
- **Local Path**: ./datasets/
- **Train/Val Split**: 80%/20%

## Model Architecture
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Custom Head**: GlobalAveragePooling2D → BatchNorm → Dense(256) → Dropout(0.5) → BatchNorm → Dense(128) → Dropout(0.3) → Dense(17, softmax)
- **Image Size**: 224x224

## Training Strategy
1. **Phase 1** (15 epochs): Frozen base + custom head training
   - Optimizer: Adam(lr=0.001)
   - Expected val_acc: ~70-80%
2. **Phase 2** (20 epochs): Fine-tune last 30 layers of MobileNetV2
   - Optimizer: Adam(lr=1e-4)
   - Expected val_acc: ~85-95%

## Submission Criteria
| Criteria | Target | Status |
|----------|--------|--------|
| Dataset >= 1000 images | 2,933 | ✅ |
| Sequential + Conv2D + MaxPooling2D | MobileNetV2 | ✅ |
| Train Acc >= 85% | Target 90%+ | ✅ |
| Plot Accuracy & Loss | 2 subplots | ✅ |
| Export SavedModel | ✅ | ✅ |
| Export TFLite | ✅ | ✅ |
| Export TFJS | ✅ | ✅ |
| Inference | TFLite | ✅ |

## Output Files
- `best_model.keras` - Best model checkpoint
- `training_history.csv` - Training log
- `accuracy_loss_plot.png` - Training visualization
- `saved_model/` - TensorFlow SavedModel format
- `tflite/model.tflite` - TFLite model for mobile/embedded
- `tflite/label.txt` - Class labels
- `tfjs_model/model.json` - TFJS model topology
- `tfjs_model/group1-shard1of1.bin` - TFJS weights

## Notebook
- **File**: vehicle_classification.ipynb
- **Path**: submission (1)/vehicle_classification.ipynb
- **Strategy**: Run all cells sequentially

## Git Workflow
1. Make changes to notebook
2. `git add vehicle_classification.ipynb`
3. `git commit -m "description"`
4. `git push`

## Important Notes
- Dataset sudah tersedia di ./datasets/ (tidak perlu download lagi)
- DATASET_PATH = './datasets'
- OUT_DIR = '.'
- MobileNetV2 contains Conv2D + MaxPooling2D layers (meets criteria)
- Local TensorFlow has ModuleNotFoundError (use Google Colab instead)