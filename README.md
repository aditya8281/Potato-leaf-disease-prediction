# Potato Leaf Disease Prediction: Deep Learning Models Comparison

## Overview

This repository contains a comparative analysis of multiple deep learning architectures for **Potato Leaf Disease (PLD) classification** across two publicly available datasets. The study evaluates the performance of different pre-trained CNN models (ResNet50, DenseNet169, DenseNet201) on both controlled and real-world potato leaf disease datasets.

## Objective

This is a baseline comparison study to:
- Implement and evaluate multiple pre-trained deep learning models for potato leaf disease classification
- Compare model performance across two distinct datasets (Mendeley and PlantVillage)
- Establish baseline metrics for understanding model behavior on structured vs. unstructured data
- Provide reproducible results through systematic experimentation with fixed random seeds
- Contribute to understanding transfer learning effectiveness in agricultural image classification

## Datasets

This analysis uses two publicly available potato leaf disease datasets:

### 1. **Mendeley Potato Disease Dataset**
   - **Direct Link**: [Mendeley Data - Potato Leaf Diseases (ptz377bwb8)](https://data.mendeley.com/datasets/ptz377bwb8/1)
   - **Classes**: 7 disease classes (Bacteria, Fungi, Healthy, Nematode, Pest, Phytophthora, Virus)
   - **Samples**: ~3,076 images
   - **Characteristics**: Real-world field images with varying lighting, backgrounds, and natural conditions
   - **Directory**: `Mendeley_Dataset/`

### 2. **PlantVillage Potato Dataset**
   - **Kaggle Link**: [PlantVillage Dataset (Kaggle)](https://www.kaggle.com/datasets/emmarex/plantdisease)
   - **Classes**: 3 disease classes (Early blight, Late blight, Healthy)
   - **Samples**: ~2,152 images
   - **Characteristics**: Controlled environment images with consistent lighting and clean backgrounds
   - **Directory**: `PlantVillage_Dataset/`

## Project Structure

```
Potato-leaf-disease-prediction/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
│
├── Mendeley_Dataset/                           # Mendeley dataset (7 classes)
│   ├── Bacteria/
│   ├── Fungi/
│   ├── Healthy/
│   ├── Nematode/
│   ├── Pest/
│   ├── Phytophthora/
│   └── Virus/
│
├── PlantVillage_Dataset/                       # PlantVillage dataset (3 classes)
│   ├── Potato___Early_blight/
│   ├── Potato___Late_blight/
│   └── Potato___healthy/
│
├── resnet50_Mendeley/                          # ResNet50 on Mendeley
│   ├── resnet50_mendeley_1.ipynb               # Run 1 (Seed: Fixed)
│   ├── resnet50_mendeley_2.ipynb               # Run 2 (Seed: Fixed)
│   ├── results_1/                              # Results from Run 1
│   │   ├── best.keras
│   │   ├── final.keras
│   │   ├── training_log.csv
│   │   ├── history.json
│   │   ├── metrics.json
│   │   ├── classification_report.txt
│   │   ├── loss_curve.png
│   │   ├── accuracy_curve.png
│   │   └── confusion_matrix.png
│   └── results_2/                              # Results from Run 2
│
├── resnet50_PlantVillage/                      # ResNet50 on PlantVillage
│   ├── resnet50_PlantVillage_1.ipynb
│   ├── resnet50_PlantVillage_2.ipynb
│   ├── results_1/
│   └── results_2/
│
├── densenet169_Mendeley/                       # DenseNet169 on Mendeley
│   ├── densenet169_mendeley_1.ipynb
│   ├── densenet169_mendeley_2.ipynb
│   ├── results_1/
│   └── results_2/
│
├── densenet169_Mendeley(only final layer)/     # DenseNet169 (Fine-tuning only final layer)
│   ├── densenet169_mendeley_1.ipynb
│   ├── densenet169_mendeley_2.ipynb
│   ├── results_1/
│   └── results_2/
│
├── densenet169_PlantVillage/                   # DenseNet169 on PlantVillage
│   ├── densenet169_plantvillage_1.ipynb
│   ├── densenet169_plantvillage_2.ipynb
│   ├── results_1/
│   └── results_2/
│
├── densenet201_Mendeley/                       # DenseNet201 on Mendeley
│   ├── densenet201_mendeley_1.ipynb
│   ├── densenet201_mendeley_2.ipynb
│   ├── results1/
│   └── results2/
│
└── densenet201_PlantVillage/                   # DenseNet201 on PlantVillage
    ├── densenet201_PlantVillage_1.ipynb
    ├── densenet201_PlantVillage_2.ipynb
    ├── results_1/
    └── results_2/
```

## Models Implemented

### 1. **ResNet50** (Residual Network with 50 layers)
   - **Architecture**: Deep residual learning framework with skip connections
   - **Pre-trained Weights**: ImageNet
   - **Input Size**: 224×224 RGB images
   - **Pooling**: Average pooling

### 2. **DenseNet169** (Densely Connected Network with 169 layers)
   - **Architecture**: Dense connections between layers promoting feature reuse
   - **Pre-trained Weights**: ImageNet
   - **Input Size**: 224×224 RGB images
   - **Variants Tested**:
     - Full fine-tuning (all layers trainable)
     - Partial fine-tuning (only final layer trainable)

### 3. **DenseNet201** (Densely Connected Network with 201 layers)
   - **Architecture**: Deeper variant of DenseNet with enhanced feature extraction
   - **Pre-trained Weights**: ImageNet
   - **Input Size**: 224×224 RGB images
   - **Pooling**: Average pooling

## Experimental Design

### Reproducibility Strategy
Each model has been trained **twice** with different random seeds to ensure:
- **Statistical robustness** of reported metrics
- **Variance analysis** of model performance
- **Consistency verification** across runs

#### Random Seed Configuration:
- **Run 1 and Run 2**: Different seed values are used for TensorFlow, NumPy, and Python's random module
- **Purpose**: Establish confidence intervals and detect potential overfitting or instability

### Training Configuration
- **Image Size**: 224×224 pixels
- **Batch Size**: 32
- **Epochs**: 100 (with early stopping)
- **Optimizer**: Adam (Learning Rate: 0.0001)
- **Loss Function**: Categorical Cross-Entropy
- **Train-Validation Split**: 80-20 (stratified)

### Data Augmentation
- Rotation range: 40°
- Zoom range: 0.3
- Horizontal flip: Enabled
- Vertical flip: Enabled
- Preprocessing: ImageNet normalization

### Callbacks Employed
- **ReduceLROnPlateau**: Reduces learning rate by 20% if validation loss plateaus (patience: 3)
- **EarlyStopping**: Stops training if validation loss doesn't improve (patience: 15, min_delta: 0.001)
- **ModelCheckpoint**: Saves best model weights based on validation loss
- **CSVLogger**: Logs training metrics for each epoch

## Results Organization

Each model training experiment generates comprehensive outputs stored in result directories:

### Output Files Structure

```
results_1/ (Run 1) / results_2/ (Run 2)
├── best.keras                  # Best model weights during training
├── final.keras                 # Final model after training completion
├── training_log.csv            # Epoch-wise training metrics
├── history.json                # Complete training history (JSON format)
├── metrics.json                # Summary metrics (Accuracy, Precision, Recall, F1, MCC, Balanced Accuracy)
├── classification_report.txt   # Detailed per-class metrics
├── loss_curve.png              # Training vs Validation Loss plot
├── accuracy_curve.png          # Training vs Validation Accuracy plot
└── confusion_matrix.png        # Confusion matrix heatmap
```

### Metrics Computed

For each model, the following evaluation metrics are calculated on the validation set:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives) - per class and weighted
- **Recall**: True positives / (True positives + False negatives) - per class and weighted
- **F1-Score**: Harmonic mean of Precision and Recall
- **Matthews Correlation Coefficient (MCC)**: Overall correlation between predictions and ground truth
- **Balanced Accuracy**: Average recall across all classes

## How to Use

### Prerequisites
```bash
pip install -r requirements.txt
```

### Running a Model Training Notebook
1. Navigate to the desired model directory (e.g., `resnet50_Mendeley/`)
2. Open the notebook (e.g., `resnet50_mendeley_1.ipynb`) in Jupyter Lab or VS Code
3. Execute all cells sequentially
4. Results will be saved in `results_1/` or `results_2/` directory

### Analyzing Results
Each experiment produces:
- **Metrics JSON**: Load with `json.load()` for quantitative analysis
- **Training History**: Visualize convergence patterns and detect overfitting
- **Confusion Matrix**: Identify problematic disease classes
- **Classification Report**: Analyze per-class performance distribution

## Model Architecture Comparison

This study evaluates three different pre-trained architectures:

| Model | Depth | Parameters | Training Time | Use Case |
|-------|-------|------------|---|---|
| ResNet50 | 50 layers | ~25M | Moderate | Balanced baseline |
| DenseNet169 | 169 layers | ~14M | Longer | Feature reuse optimization |
| DenseNet201 | 201 layers | ~20M | Longest | Deeper feature extraction |

All models are trained with identical configurations for fair comparison.

## Related Work & References

This analysis is informed by recent advances in deep learning for agricultural disease detection. A notable reference is the PLDNet paper which explores hybrid CNN-Transformer architectures for potato disease classification:

- **"A hybrid CNN-transformer model with adaptive activation function for potato leaf disease classification"**
  - Published in Scientific Reports (2026) 16:4282
  - Link: [https://www.nature.com/articles/s41598-025-34406-4](https://www.nature.com/articles/s41598-025-34406-4)

However, this repository focuses on evaluating baseline models (ResNet50, DenseNet169, DenseNet201) as a foundation study rather than implementing advanced hybrid architectures.

### Study Outcomes

This comparison provides:
1. **Model performance metrics** on controlled (PlantVillage) and real-world (Mendeley) datasets
2. **Insights into generalization** across different data distributions
3. **Baseline results** from multiple runs for consistency validation
4. **Practical understanding** of which models work better in different scenarios

## Potential Next Steps

- Try ensemble methods combining multiple models
- Test-time augmentation strategies
- Evaluate other pre-trained architectures (EfficientNet, MobileNetV3)
- Implement advanced models from recent literature (PLDNet, hybrid architectures)
- Deploy trained models for real-world field testing

## Requirements

See `requirements.txt` for complete dependency list. Key packages:
- **TensorFlow/Keras** (2.10+): Deep learning framework
- **scikit-learn**: Machine learning utilities and metrics
- **pandas**: Data manipulation
- **matplotlib & seaborn**: Visualization
- **NumPy**: Numerical computing
- **GPU support**: CUDA (optional but recommended for faster training)

## Citation

If you use this repository or datasets in your work, please cite the original datasets:

```bibtex
@dataset{mendeley_potato_2024,
  title={Potato Leaf Disease Dataset},
  author={Shabrina, N. H. et al.},
  url={https://data.mendeley.com/datasets/ptz377bwb8/1},
  year={2024}
}

@dataset{plantvillage_2015,
  title={PlantVillage Dataset},
  url={https://www.kaggle.com/datasets/emmarex/plantdisease},
  year={2015}
}
```

## Contact & Support

For questions regarding this reproducibility study, refer to the individual notebook files for detailed implementation notes and comments.

---

**Study Completion**: May 2026  
**Version**: 1.0  
**Status**: Complete Implementation with Comparative Analysis
