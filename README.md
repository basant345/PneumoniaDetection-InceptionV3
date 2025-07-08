# PneumoniaDetection-InceptionV3

This project aims to detect **Pneumonia** in chest X-ray images using a deep learning approach. I used **InceptionV3**, a pre-trained model, and applied **transfer learning** on the **PneumoniaMNIST** dataset.

---

##  Objective

The goal is to fine-tune InceptionV3 to classify chest X-ray images into:
- **0** = Normal (No Pneumonia)
- **1** = Pneumonia (Presence of Pneumonia)

I used several deep learning techniques to improve the model’s performance and generalization.

---

##  Dataset

- **Dataset Name**: PneumoniaMNIST  
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/rijulshr/pneumoniamnist)
- **Format**: `.npz` file with 3 pre-split sets:
  - `train_images`, `train_labels`
  - `val_images`, `val_labels`
  - `test_images`, `test_labels`
- **Image Size**: 28x28 grayscale (converted to 299x299 RGB for InceptionV3)

---

##  Methodology

###  Data Preprocessing

- Converted grayscale X-ray images to RGB (3 channels).
- Resized images to 299x299 as required by InceptionV3.
- Normalized pixel values to range [0, 1].

###  Transfer Learning (Model Setup)

- **Base Model**: InceptionV3 (ImageNet pre-trained, `include_top=False`)
- **Custom Classifier Head**:
  - `GlobalAveragePooling2D`
  - `Dropout(0.5)` to reduce overfitting
  - `Dense(128, activation='relu')`
  - `Dense(2, activation='softmax')` for binary classification

###  Handling Class Imbalance

- Used `class_weight` from `sklearn` to give higher importance to the minority class (Normal).
- This prevents the model from being biased toward the majority class (Pneumonia).

###  Overfitting Prevention

- **Data Augmentation**: Rotation, zoom, shift, and flip using `ImageDataGenerator`.
- **Dropout Layer**: 50% dropout added in the custom head.
- **Early Stopping**: Stops training if validation loss does not improve for 5 consecutive epochs.

---

##  Evaluation Metrics

I used the following performance metrics to evaluate the model:

- **Accuracy**: Overall correct predictions
- **Precision**: How many predicted pneumonia cases were actually pneumonia
- **F1 Score**: Balance between precision and recall
- **AUC (ROC Curve)**: Measures model’s ability to separate classes
- **Confusion Matrix**: Shows TP, FP, TN, FN counts

---

##  Hyperparameter

-**Learning Rate**: 1e-4 – Fine-tunes the pre-trained model slowly to avoid losing useful features.
-**Batch Size**: 32 – Balanced choice for stable training and efficient memory usage.
-**Epochs**: 20 – Capped training length with EarlyStopping to avoid overfitting.
-**Dropout Rate**: 0.5 – Helps prevent overfitting by randomly dropping neurons.
-**Image Size**: 299x299 – Required input size for InceptionV3.
-**Optimizer**: Adam – Adaptive and efficient for transfer learning tasks.

---

##  Results

| Metric     | overall |
|------------|-----------------|
| F1 Score   | 0.89            |
| Precision  | 0.83            |
| AUC Score  | 0.94            |

