# Wafex: Semiconductor Wafer Defect Classification

Deep Learning model for automated classification of semiconductor wafer defects using Convolutional Neural Networks (CNN).

## Project Overview

This project implements a CNN-based classifier to identify and categorize defect patterns in semiconductor wafer maps. Using the WM-811K dataset containing real manufacturing data, the model achieves 93.14% accuracy across 8 defect types.

### Problem Statement

In semiconductor manufacturing, wafer defects cost billions annually. Manual inspection is slow, inconsistent, and prone to human error. This automated ML solution enables:
- Fast, consistent defect screening
- Early detection of manufacturing issues
- Yield optimization through pattern recognition

## Dataset

**WM-811K (LSWMD) Dataset**
- Source: [Kaggle - WM811K Wafer Map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
- 811,457 wafer maps from real semiconductor fabrication
- 8 defect pattern classes
- Spatial defect distribution data

**Defect Classes:**
1. Center
2. Donut
3. Edge-Loc
4. Edge-Ring
5. Loc
6. Near-full
7. Random
8. Scratch

## Results

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 93.14% |
| **Test Loss** | 0.1914 |
| **Macro Avg Precision** | 0.9100 |
| **Macro Avg Recall** | 0.8973 |
| **Macro Avg F1-Score** | 0.9031 |
| **Improvement over Baseline** | +3.78% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Center | 0.9453 | 0.9849 | 0.9647 | 859 |
| Donut | 0.8684 | 0.8919 | 0.8800 | 111 |
| Edge-Loc | 0.9034 | 0.8921 | 0.8977 | 1038 |
| Edge-Ring | 0.9809 | 0.9819 | 0.9814 | 1936 |
| Loc | 0.8545 | 0.8498 | 0.8522 | 719 |
| Near-full | 0.9310 | 0.9000 | 0.9153 | 30 |
| Random | 0.9545 | 0.8497 | 0.8991 | 173 |
| Scratch | 0.8419 | 0.8277 | 0.8347 | 238 |

## Model Architecture

**Improved CNN Architecture:**
- **Input:** 64x64 grayscale images
- **Conv Block 1:** 2x Conv2D(32) + MaxPooling + BatchNorm + Dropout(0.25)
- **Conv Block 2:** 2x Conv2D(64) + MaxPooling + BatchNorm + Dropout(0.25)
- **Conv Block 3:** 2x Conv2D(128) + MaxPooling + BatchNorm + Dropout(0.25)
- **Dense Layers:** 256 → Dropout(0.5) → 128 → Dropout(0.5) → 8 (Softmax)
- **Total Parameters:** ~2.5M

**Training Configuration:**
- Optimizer: Adam (lr=0.001)
- Loss: Sparse Categorical Crossentropy
- Batch Size: 64
- Epochs: 15 (with Early Stopping)
- Callbacks: EarlyStopping, ReduceLROnPlateau

## Tech Stack

- **Framework:** TensorFlow 2.18 / Keras
- **Data Processing:** NumPy, Pandas, Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Python 3.11, Kaggle Notebooks

## Project Structure

```
wafex/
│
├── wafer_defect_classification.ipynb    # Main notebook
├── README.md                             # Project documentation
├── requirements.txt                      # Dependencies
│
└── outputs/
    ├── wafer_cnn_improved.h5            # Trained model
    ├── sample_wafer_defects.png         # Sample defect patterns
    ├── confusion_matrix.png             # Confusion matrix
    ├── per_class_performance.png        # Per-class metrics
    ├── prediction_samples.png           # Model predictions
    ├── dashboard_part1.png              # Training dashboard
    ├── f1_scores.png                    # F1 scores by class
    └── performance_summary.png          # Summary table
```

## Getting Started

### Prerequisites

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/saagarnkashyap/wafex.git
cd wafex
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset:
   - Visit [Kaggle WM-811K Dataset](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)
   - Download `LSWMD.pkl`
   - Place in `data/` directory

### Usage

1. **Train the model:**
   ```python
   # Run the Jupyter notebook
   jupyter notebook wafer_defect_classification.ipynb
   ```

2. **Load pre-trained model:**
   ```python
   from tensorflow import keras
   model = keras.models.load_model('outputs/wafer_cnn_improved.h5')
   ```

3. **Make predictions:**
   ```python
   # Preprocess your wafer map to 64x64
   prediction = model.predict(wafer_map)
   defect_class = np.argmax(prediction)
   ```

## Key Features

- Handles imbalanced classes with stratified splitting
- Robust preprocessing pipeline for legacy pickle format
- Comprehensive evaluation metrics and visualizations
- Production-ready model saved in H5 format
- Detailed per-class performance analysis
- Training history and comparison dashboards

## Insights

**Defect Pattern Recognition:**
- **Edge-Ring defects** (98.1% F1): Highest accuracy - likely etching or deposition issues
- **Center defects** (96.5% F1): Strong performance - coating or alignment problems
- **Scratch defects** (83.5% F1): Most challenging - irregular patterns harder to classify

**Model Behavior:**
- Early stopping triggered after 15 epochs
- Learning rate reduction improved convergence
- Deeper architecture significantly outperformed baseline

## Learnings

1. **Data Preprocessing Matters:** Legacy pickle compatibility required careful handling
2. **Architecture Depth:** Deeper networks (3 conv blocks) outperformed shallow baseline by 3.78%
3. **Regularization Impact:** BatchNorm + Dropout prevented overfitting on imbalanced classes
4. **Domain Knowledge:** Understanding semiconductor defects helped interpret model predictions

## Future Improvements

- Increase training samples to 100K+ for better generalization
- Implement data augmentation (rotation, flip) for minority classes
- Experiment with ResNet/EfficientNet architectures
- Add attention mechanisms to highlight defect regions
- Deploy as REST API for real-time inference
- Multi-label classification for overlapping defects

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- WM-811K dataset by MIR Lab, National Taiwan University
- Kaggle for hosting the dataset and compute resources
- Semiconductor manufacturing domain experts for defect insights

## Contact

**Saagar N Kashyap**
- GitHub: [@saagarnkashyap](https://github.com/saagarnkashyap)
- LinkedIn: [Saagar N Kashyap](https://www.linkedin.com/in/saagar-n-kashyap/)
- Email: saagarcourses@gmail.com


