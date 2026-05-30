# Neural Digit Recognizer 🔢

An advanced deep learning system for handwritten digit recognition, built on MNIST dataset using TensorFlow/Keras. Features cutting-edge CNN architecture with residual connections, squeeze-and-excitation attention, and multi-format export capabilities.

## ✨ Features

- **Advanced CNN Architecture**
  - Residual (skip) connections for improved gradient flow
  - Squeeze-and-Excitation (SE) channel attention mechanism
  - Data augmentation layers (rotation, zoom, translation, contrast)
  - Global Average Pooling for efficient spatial reduction

- **Training Enhancements**
  - Mixup augmentation for better generalization
  - Label smoothing for improved model calibration
  - Learning rate scheduling with early stopping
  - Ensemble model support for higher accuracy

- **Multiple Export Formats**
  - Keras model (.keras)
  - TensorFlow Lite (.tflite) - standard and quantized versions
  - ONNX format support for cross-platform deployment

- **Web Interface & API**
  - Flask REST API for predictions
  - Interactive web UI for drawing digits
  - Multi-digit recognition with automatic segmentation
  - Single and batch prediction endpoints

- **Comprehensive Evaluation**
  - Full test set evaluation with detailed metrics
  - Confusion matrix generation
  - Per-class performance analysis
  - Prediction visualization

## 🚀 Quick Start

### Installation

1. **Clone/Download the repository**
   ```bash
   cd "Neural Digit Recognizer"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # or
   source venv/bin/activate  # macOS/Linux
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Train a Model

```bash
# Train with default settings
python main.py

# Train with custom hyperparameters
python main.py --epochs 30 --batch-size 128 --initial-lr 0.001

# Train an ensemble of 5 models
python main.py --use-ensemble --ensemble-size 5

# Ablation study: disable residual connections and SE attention
python main.py --no-residual --no-se
```

### Evaluate the Model

```bash
# Full evaluation on test set
python evaluate.py

# Generates confusion matrix and detailed metrics
```

### Launch Web Server

```bash
# Start Flask server
python server.py

# Visit http://localhost:5000 in your browser
# Draw a digit and get predictions in real-time
```

### Export Models

```bash
# Export to all formats (Keras, TFLite standard, TFLite quantized, ONNX)
python export.py

# Exports saved to saved_model/
```

## 📁 Project Structure

```
Neural Digit Recognizer/
├── main.py                 # Training pipeline
├── model.py                # CNN model definition
├── evaluate.py             # Evaluation and metrics
├── export.py               # Multi-format export
├── server.py               # Flask web API
├── config.py               # Configuration management
├── utils.py                # Helper functions
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Project metadata
├── saved_model/            # Trained models
│   ├── mnist_advanced.keras
│   ├── mnist_advanced.tflite
│   ├── mnist_advanced_int8.tflite
│   └── mnist_advanced_fp16.tflite
├── static/                 # Web server assets
├── templates/
│   └── index.html          # Web UI
├── results/                # Evaluation results
├── experiments/            # Training logs and configs
└── logs/                   # TensorBoard logs
```

## 🏗️ Model Architecture

```
Input (28×28×1)
    ↓
Data Augmentation (rotation, zoom, translation, contrast)
    ↓
[Feature Extraction × 3]
  • Conv2D → BatchNorm → ReLU
  • Residual Block (2× Conv2D with skip connection)
  • Squeeze-and-Excitation Attention
  • MaxPooling
    ↓
Global Average Pooling
    ↓
Dense(256) → Dropout(0.5) → BatchNorm
    ↓
Dense(128) → Dropout(0.3) → BatchNorm
    ↓
Dense(10, softmax)
    ↓
Output (digit probabilities)
```

## ⚙️ Configuration

All hyperparameters are centralized in `config.py`:

### Model Architecture
- `filters`: Feature map counts per block `[32, 64, 128]`
- `dense_units`: Dense layer sizes `[256, 128]`
- `dropout_rates`: Dropout rates `[0.5, 0.3]`
- `use_residual`: Enable skip connections `True`
- `use_se_attention`: Enable SE attention `True`

### Data Augmentation
- `augment_rotation`: Rotation range `0.1` (±10%)
- `augment_zoom`: Zoom range `0.1` (±10%)
- `augment_translation`: Translation `0.1` (±10%)
- `augment_contrast`: Contrast range `0.1` (±10%)
- `use_mixup`: Enable mixup augmentation `True`
- `mixup_alpha`: Mixup parameter `0.2`

### Training
- `epochs`: Number of training epochs `20`
- `batch_size`: Batch size `64`
- `initial_lr`: Starting learning rate `1e-3`
- `label_smoothing`: Label smoothing value `0.1`
- `val_split`: Validation split ratio `0.2`

### Export
- `export_tflite`: Export TensorFlow Lite `True`
- `export_onnx`: Export ONNX format `True`
- `tflite_quantize`: Quantize TFLite models `True`

## 📊 API Reference

### Web Server Endpoints

#### GET `/`
- Returns the web interface HTML

#### POST `/predict`
- **Description**: Single digit prediction
- **Request**:
  ```json
  {
    "image": "<base64_encoded_image>"
  }
  ```
- **Response**:
  ```json
  {
    "digit": 7,
    "confidence": 0.9987,
    "probabilities": [0.0001, 0.0, ..., 0.9987]
  }
  ```

#### POST `/predict-multi`
- **Description**: Multi-digit recognition with automatic segmentation
- **Request**:
  ```json
  {
    "image": "<base64_encoded_image>"
  }
  ```
- **Response**:
  ```json
  {
    "digits": [3, 7, 2],
    "confidence_scores": [0.999, 0.998, 0.997],
    "num_digits": 3
  }
  ```

## 🔍 Performance Metrics

Models are evaluated on the MNIST test set (10,000 samples):

| Model | Accuracy | F1-Score | Size (MB) | Inference Time (ms) |
|-------|----------|----------|-----------|-------------------|
| Keras (.keras) | ~99.5% | 0.995 | 2.1 | 15-20 |
| TFLite (standard) | ~99.4% | 0.994 | 0.8 | 8-12 |
| TFLite (int8) | ~99.2% | 0.992 | 0.4 | 5-8 |
| TFLite (fp16) | ~99.4% | 0.994 | 1.2 | 6-10 |

*Benchmarks are approximate and vary based on hardware*

## 📈 Training Monitoring

Monitor training progress using TensorBoard:

```bash
tensorboard --logdir=logs/tensorboard
```

Then visit `http://localhost:6006` in your browser to view:
- Training/validation loss and accuracy curves
- Learning rate schedule
- Gradient histograms
- Data augmentation effects

## 🔧 Advanced Usage

### Custom Data

To train on custom digit images:

1. Prepare 28×28 grayscale images
2. Modify the data loading section in `main.py`
3. Ensure labels are in range [0-9]

### Batch Inference

```python
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('saved_model/mnist_advanced.keras')

# Prepare batch of images (batch_size, 28, 28, 1)
predictions = model.predict(image_batch)
digits = np.argmax(predictions, axis=1)
```

### Model Ensemble

```bash
python main.py --use-ensemble --ensemble-size 5
```

Individual ensemble members are saved as:
- `mnist_ensemble_0.keras`
- `mnist_ensemble_1.keras`
- etc.

## 🐛 Troubleshooting

**Model not found when running server**
- Ensure you've run `python main.py` first to train and save the model
- Check that `saved_model/mnist_advanced.keras` exists

**Out of memory errors**
- Reduce `batch_size` in config.py
- Disable data augmentation temporarily
- Use TFLite quantized version

**Poor accuracy on custom images**
- Ensure images are centered, 28×28, grayscale
- Images should be normalized to [0, 1]
- Try preprocessing with higher contrast

## 📦 Dependencies

- **TensorFlow** (≥2.20.0): Deep learning framework
- **NumPy** (≥2.4.1): Numerical computing
- **Scikit-learn** (≥1.4.0): ML utilities and metrics
- **Matplotlib** (≥3.10.8): Visualization
- **Flask** (≥3.0.0): Web server
- **Flask-CORS** (≥4.0.0): Cross-origin support
- **Gradio** (≥4.0.0): Alternative UI option
- **TensorBoard** (≥2.20.0): Training visualization
- **PyYAML** (≥6.0): Config management
- **SciPy** (≥1.11.0): Scientific computing

## 📝 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report issues
- Suggest improvements
- Submit pull requests
- Share results and benchmarks

## 📚 References

- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [Residual Networks](https://arxiv.org/abs/1512.03385)
- [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)
- [Mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)

---

**Last Updated**: May 2026  
**Python Version**: 3.9+  
**Framework**: TensorFlow 2.20+
