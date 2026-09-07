# Automated-Dermoscopic-Image-Classification-with-CNNs-for-melanoma-detection
CNN based classification of dermoscopic images into benign lesions and malignant melanoma. The project compares VGG16, ResNet50 and DenseNet121 using transfer learning, 3-fold cross-validation, targeted optimisation, quantitative metrics and Grad-CAM explainability.



## Project Overview
Melanoma is an aggressive type of skin cancer for which early diagnosis is a major prognostic factor. However, distinguishing malignant melanoma from benign skin lesions is challenging because dermoscopic images may show substantial visual variability and overlapping clinical features. In this context, deep learning methods can support image analysis by learning visual patterns directly from dermoscopic images.

This repository contains the code, results, visualisations, and supporting material for the Bachelor’s thesis: **Automated Classification of Dermoscopic Images Using CNNs: Deep Learning Systems for Melanoma Diagnosis**
This project investigates the use of convolutional neural networks (CNNs) for the binary classification of dermoscopic images into **benign lesions** and **malignant melanoma**. The work was developed as a Bachelor’s Thesis in Biomedical Engineering and was implemented in Python using TensorFlow and Keras within Google Colab. 

The study compares three ImageNet-pretrained CNN architectures:

- **VGG16**, a sequential convolutional architecture;
- **ResNet50**, based on residual connections;
- **DenseNet121**, based on dense feature reuse.

The purpose was not only to identify the model with the best quantitative performance, but also to investigate how preprocessing choices, transfer-learning strategy, regularisation, and fine-tuning influence model behaviour in a dermoscopic classification task.

The experimental workflow consists of two main phases:
1. **Baseline comparison**  
   VGG16, ResNet50, and DenseNet121 were trained using the same transfer-learning pipeline.
2. **Targeted optimisation**  
   ResNet50, VGG16, and DenseNet121 were optimised using model-specific or regularisation-oriented strategies.

Grad-CAM was subsequently used to qualitatively inspect which image regions contributed most to each model prediction.


## Objectives
Main objectives of the project:

1. **Build a reproducible melanoma classification pipeline**  
   Train CNN-based binary classifiers on a public dermoscopic image dataset containing benign and malignant lesions.

2. **Compare three CNN architectures under common conditions and evaluate robustness through stratified 3-fold cross-validation**  
   VGG16, ResNet50, and DenseNet121 were first evaluated through a common baseline pipeline. The models used ImageNet weights, frozen convolutional backbones, the same classification head, image resizing to 224 × 224 pixels, data augmentation, and the same training configuration.
Each architecture was trained and evaluated across three stratified folds. Performance was assessed using accuracy, sensitivity, specificity, precision, F1-score, AUC-ROC, ROC curves, and confusion matrices. Results are reported as descriptive mean values across folds.

3. **Optimise model configurations and evaluate robustness through stratified 3-fold cross-validation**  
   Following the baseline results, targeted optimisation strategies were applied. ResNet50 received model-specific preprocessing, partial fine-tuning, a lower learning rate, and early stopping. VGG16 and DenseNet121 were optimised through stronger augmentation, L2 regularisation, Batch Normalization, increased Dropout, EarlyStopping, and learning-rate reduction on plateau. Performance was assessed using accuracy, sensitivity, specificity, precision, F1-score, AUC-ROC, ROC curves, and confusion matrices. Results are reported as descriptive mean values across folds.

4. **investigate explainability with Grad-CAM**
     Grad-CAM was then used to qualitatively inspect the image regions most associated with model predictions.


## Main Findings

In the baseline configuration, DenseNet121 achieved the best overall balance between sensitivity and specificity. VGG16 showed relatively high sensitivity but lower specificity. ResNet50 initially performed poorly because it classified almost all samples as benign, resulting in near-zero sensitivity despite very high specificity.

After targeted optimisation, ResNet50 showed the largest improvement and achieved the highest mean values among the tested configurations for accuracy, sensitivity, F1-score, and AUC. These findings highlight that performance depends not only on the selected CNN architecture, but also on input preprocessing, the transfer-learning strategy, regularisation, and the extent to which the pretrained backbone is adapted to the new domain.

This repository is intended for educational and research purposes. The models are not clinically validated and must not be used as diagnostic tools.


## Dataset

A public dermoscopic-image dataset obtained from Kaggle was used. The task is binary classification:

|Class folder|Numerical label|Meaning|
|-|-:|-|
|`benign`|0|Benign skin lesion|
|`malignant`|1|Malignant melanoma|

The dataset is organized as follows:

- **Training set:** 2,637 images
- **Test set:** 660 images

The training images were used for model fitting and stratified cross-validation. The test directory was kept separate from cross-validation and was used for the first training/test runs and for the qualitative Grad-CAM analysis.

> The image files themselves are not redistributed in this repository. 
The dataset can be found at this link: https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign
Please obtain the dataset from its original Kaggle source and arrange it according to the directory structure below.

```text
kaggle/
├── train/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```
\---

Class labels are inferred from folder names:

```text
benign    -> 0
malignant -> 1
```


## Phase 1 - Baseline Pipeline
The baseline was designed to compare the three architectures under the same experimental conditions.

### Common settings
- Input image size: `224 × 224 × 3`
- Batch size: `32`
- Maximum training epochs: `10`
- Optimizer: `Adam`
- Loss function: `binary_crossentropy`
- Training metric: `accuracy`
- Classification threshold: `0.5`
- Best model selection: minimum validation loss using `ModelCheckpoint`

### Preprocessing and augmentation
All images are resized to `224 × 224` pixels and initially normalised to the `[0, 1]` range.

The training pipeline included:
- Random horizontal flip
- Random rotation: `0.2`
- Random zoom: `0.1`

### Transfer learning setup
For the baseline, all backbones were loaded with ImageNet weights and used as frozen feature extractors:

```python
weights="imagenet"
include_top=False
base_model.trainable = False
```

The same classification head was added to every architecture:

```text
Backbone CNN
→ GlobalAveragePooling2D
→ Dense(128, ReLU)
→ Dropout(0.5)
→ Dense(1, sigmoid)
```

---

## 3-Fold Cross-Validation
After verifying the baseline implementation, the models were evaluated using stratified 3-fold cross-validation.

- `StratifiedKFold(n_splits=3, shuffle=True, random_state=42)`
- Each fold preserves approximately the same benign/malignant class proportion.
- For each fold, two partitions are used for training and one for validation.
- Each architecture is trained independently for each fold.
- Data augmentation is applied only to the training portion of each fold.
- `EarlyStopping` is used during cross-validation to reduce unnecessary training and limit overfitting.

The reported cross-validation results are descriptive means across three folds. Because only three folds were used, the repository does not claim statistical significance between architectures.

---

## Model Optimisation

### ResNet50 optimisation

ResNet50 showed critical baseline behavior, with predictions strongly biased toward the benign class. The optimized configuration included:

- ResNet-specific ImageNet preprocessing through `preprocess_input` instead of only `x / 255` normalization
- Partial fine-tuning of the final 30 backbone layers of the pretrained backbone
- Reduced Adam learning rate: `1e-4`
- `EarlyStopping` with patience `3` and `restore\_best\_weights=True`
- Best checkpoint selection based on validation loss
- 

### VGG16 and DenseNet121 optimisation

VGG16 and DenseNet121 were jointly optimised using regularisation and training-control techniques:

- Stronger random zoom: `0.2`
- L2 regularisation on the Dense layer: `l2(0.001)`
- Batch Normalization after the Dense layer
- Increased Dropout: `0.6`
- `EarlyStopping` with patience `5`
- `ReduceLROnPlateau`:
  - monitored metric: validation loss
  - factor: `0.5`
  - patience: `2`
  - minimum learning rate: `1e-6`

The original backbone architecture of VGG16 and DenseNet121 was preserved.


The optimized models were also evaluated using stratified 3-fold cross-validation.

## Evaluation Metrics

Models were evaluated using the following metrics:

- **Accuracy**
- **Sensitivity**
- **Specificity**
- **F1-score**
- **AUC-ROC**
- **Confusion matrix**
- **ROC curve**

For the binary class convention used in the project:

|Actual class|Predicted benign|Predicted malignant|
|-|-:|-:|
|Benign|True Negative (TN)|False Positive (FP)|
|Malignant melanoma|False Negative (FN)|True Positive (TP)|


For melanoma classification, sensitivity is particularly relevant because false negatives correspond to malignant lesions incorrectly classified as benign.

---

## Results Summary
Performance was summarized as the arithmetic mean across the three validation folds. The results are descriptive: with only three folds, no formal inferential statistical comparison or claim of statistically significant superiority was made.

### Baseline: mean 3-fold cross-validation results

| Model | Accuracy | Sensitivity | Specificity | F1-score | AUC |
|---|---:|---:|---:|---:|---:|
| VGG16 | 0.8153 | 0.8688 | 0.7708 | 0.8102 | 0.8998 |
| ResNet50 | 0.5465 | 0.0033 | 0.9979 | 0.0066 | 0.7741 |
| DenseNet121 | 0.8479 | 0.8454 | 0.8500 | 0.8346 | 0.9363 |

DenseNet121 achieved the strongest and most balanced baseline performance. VGG16 achieved the highest baseline sensitivity but lower specificity. ResNet50 showed a degenerate classification pattern: it predicted nearly all validation samples as benign, which yielded very high specificity but almost zero sensitivity and F1-score.

### Optimised models: mean 3-fold cross-validation results

| Model | Accuracy | Sensitivity | Specificity | F1-score | AUC |
|---|---:|---:|---:|---:|---:|
| VGG16 | 0.8244 | 0.8279 | 0.8215 | 0.8089 | 0.9109 |
| DenseNet121 | 0.8589 | 0.8638 | 0.8549 | 0.8474 | 0.9364 |
| ResNet50 | 0.8786 | 0.8972 | 0.8632 | 0.8705 | 0.9516 |

The largest improvement was observed for ResNet50. In the optimized configuration, ResNet50 achieved the highest mean accuracy, sensitivity, F1-score, and AUC among the evaluated configurations. DenseNet121 remained strong and well balanced in both phases. VGG16 improved in accuracy, specificity, and AUC, while its mean sensitivity and F1-score were slightly lower than in the baseline configuration.

---

## Grad-CAM Explainability
Grad-CAM was used as a qualitative post-hoc explainability method. It highlights image regions that positively contribute to the model output for the malignant class.

The analysis was performed on the first 10 image files, ordered alphabetically by filename, from each test-set class:
- 10 benign test images
- 10 malignant test images
- A total of 20 images per model

The following final convolutional layers were used:

| Model | Grad-CAM layer |
|---|---|
| VGG16 | `block5_conv3` |
| ResNet50 | `conv5_block3_out` |
| DenseNet121 | `conv5_block16_2_conv` |

For each image, the pipeline saves:
- the original image;
- the normalized Grad-CAM heatmap;
- an image/heatmap overlay with `alpha=0.4`;
- a three-panel visualization containing original image, heatmap, and overlay;
- a CSV row with filename, prediction, probability of malignancy, TP/TN/FP/FN outcome, selected convolutional layer, and output paths.

**Implementation note**
The optimized ResNet50 model was trained using the ResNet-specific `preprocess\_input` function. Any inference or Grad-CAM script used for this optimized model should apply the same preprocessing before prediction and gradient computation. Using simple `\[0, 1]` rescaling instead would not be fully consistent with the training input pipeline.


Qualitatively, DenseNet121 generally produced maps aligned with the lesion region, VGG16 showed more diffuse activation, and optimized ResNet50 produced richer and less repetitive activation maps than its baseline version. Grad-CAM maps are not lesion segmentations and do not establish clinical validity; they should be interpreted as exploratory visual explanations.
Grad-CAM is used as an exploratory qualitative method. The heatmaps do not represent lesion segmentation and do not independently establish clinical validity.

---

## Repository Structure

```text
.
├── README.md
├── notebook/
│   ├── 01_baseline_training&gradcam.ipynb
│   ├── 02_baseline_3fold_cross_validation.ipynb
│   ├── 03_ResNet50_optimization_training&gradcam.ipynb
│   ├── 04_optimized_ResNet50_3fold_cross_validation.ipynb
│   ├── 05_Vgg16_densenet121_optimization_training&gradcam.ipynb
│   └── 06_optimized_Vgg16_DenseNet121_3fold_cross_validation.ipynb
│
├── Results/
└── GradCam/
    ├── baseline_models/
    │   ├── all_models_gradcam_results.csv
    │   ├── benign_grid_comparison.png
    │   ├── comparison_summary.csv
    │   ├── malignant_grid_comparison.png
    │   └── vgg16_vs_resnet_vs_densenet_comparison.csv
    │
    ├── optimized_models/
    │   ├── OPTResnet/
    │   │   ├── benign_01_panel.png
    │   │   ├── benign_02_panel.png
    │   │   ├── benign_03_panel.png
    │   │   ├── benign_04_panel.png
    │   │   ├── benign_05_panel.png
    │   │   ├── benign_06_panel.png
    │   │   ├── benign_07_panel.png
    │   │   ├── benign_08_panel.png
    │   │   ├── benign_09_panel.png
    │   │   ├── benign_10_panel.png
    │   │   ├── malignant_01_panel.png
    │   │   ├── malignant_02_panel.png
    │   │   ├── malignant_03_panel.png
    │   │   ├── malignant_04_panel.png
    │   │   ├── malignant_05_panel.png
    │   │   ├── malignant_06_panel.png
    │   │   ├── malignant_07_panel.png
    │   │   ├── malignant_08_panel.png
    │   │   ├── malignant_09_panel.png
    │   │   ├── malignant_10_panel.png
    │   │   ├── benign_grid_comparison.png
    │   │   ├── malignant_grid_comparison.png
    │   │   ├── resnet_optimized_gradcam_results.csv
    │   │   └── resnet_optimized_summary.csv
    │   │
    │   ├── VggDenseNet_benign_grid_comparison.png
    │   ├── VggDenseNet_gradcam_results.csv
    │   ├── VggDenseNet_malignant_grid_comparison.png
    │   └── vgg16_vs_densenet_comparison.csv
    │
    ├── baseline-first phase.zip
    └── optimized-second phase.zip
```
## Trained model files

The trained Keras model checkpoints are not included in this repository because `.keras` files are large binary files and would significantly increase the repository size.

The repository therefore contains:

- the complete training and evaluation notebooks;
- the Grad-CAM outputs;
- the comparison grids;
- the CSV files containing predictions and summaries;
- the compressed result collections for the baseline and optimized phases.

The ImageNet-pretrained backbones are loaded automatically by Keras when the notebooks are executed. The task-specific models can be regenerated by running the corresponding notebooks with the original dataset and the required folder structure.

This choice keeps the repository lightweight while preserving the code and selected outputs needed to understand and reproduce the experimental workflow.


During the original Colab experiments, outputs were stored in Google Drive folders such as:

```text
/content/drive/MyDrive/results10
/content/drive/MyDrive/CrossValidation/OldModels
/content/drive/MyDrive/resultsResnetOPT1
/content/drive/MyDrive/resultsVgg16DensenetOPT1
/content/drive/MyDrive/GradCam/OldModels
/content/drive/MyDrive/GradCam/OPTResnet
/content/drive/MyDrive/GradCam/OPTvgg16andDensenet
```

The outputs include Keras model checkpoints (`.keras`), training histories (`.json`), fold-level metrics (`.csv`), summary metrics (`.csv`), confusion matrices, ROC curves, learning curves, metric plots, and Grad-CAM panels (`.png`).

\---

---

## Installation
The experiments were developed in Google Colab. For local use, install a compatible Python environment and the required libraries:

```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
```

The core dependencies are:

```text
TensorFlow / Keras
NumPy
Pandas
Matplotlib
Seaborn
scikit-learn
```

If using Google Colab, mount Google Drive and update the project paths before executing the notebooks:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Set the dataset and output locations in the configuration section of each notebook, for example:

```python
TRAIN\_DIR = "/content/drive/MyDrive/kaggle/train"
TEST\_DIR = "/content/drive/MyDrive/kaggle/test"
BASE\_PATH = "/content/drive/MyDrive/your\_results\_folder"
```

\---

## Reproducibility Notes

- Random seed used for cross-validation: `42`
- Models and training histories are saved after training.
- The best checkpoint is selected according to validation loss.
- Cross-validation results are reported as means across three folds.
- The dataset itself is not included in this repository; it must be downloaded separately from its original Kaggle source.
- Paths in the original Colab code use Google Drive and should be changed when running locally.

---

## Limitations

This repository is intended for research and educational purposes only.

- The models are not validated for clinical deployment.
- Results were obtained with a public dataset and may not generalise to external populations, imaging devices, or clinical settings.
- Cross-validation used three folds; results are descriptive and do not demonstrate statistically significant superiority between models.
- Grad-CAM heatmaps provide qualitative explanations and cannot be treated as clinically validated lesion segmentations.
- A dedicated external validation dataset and broader clinical evaluation would be required before considering real-world use.

Potential future work includes validation on larger and external datasets, repeated or higher-fold cross-validation, class-balancing strategies, threshold optimization, prospective clinical evaluation, and expert assessment of explainability maps.

---

## Citation

If you use or refer to this repository, please cite the associated thesis:

```text
Sacilotto, G. (2026).
Automated Classification of Dermoscopic Images Using CNNs:
Deep Learning Systems for Melanoma Diagnosis.
Bachelor’s Thesis in Biomedical Engineering.
```

---

## License and data use

The code in this repository is intended for academic and educational use. The dataset remains subject to its original Kaggle license, terms of use, and attribution requirements. Do not upload or redistribute the dataset unless its original license explicitly permits doing so.




