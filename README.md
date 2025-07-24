📄 README.md 

# 🧠 Human Detection Using HOG + SVM, ANN, and CNN

This project presents a complete human detection system using three different models:  
- **SVM** with HOG features  
- **ANN (MLP)** using flattened grayscale  
- **CNN** using deep convolutional layers

The goal is to detect whether a given image contains a human, and to **compare classical ML and deep learning** approaches for this task.

---

## 📁 Project Structure

├── data/
│ ├── pos_person/ # Human images
│ ├── neg_person/ # Non-human images
│ └── predictTestImages/ # Test images for predictions
├── model/
│ ├── svm_model.pkl
│ ├── human_detector_ann.h5
│ └── human_detector_cnn.h5
├── notebooks/
│ ├── 1_train_svm.ipynb # Train SVM with HOG features
│ ├── 2_train_ann.ipynb # Train ANN (MLP)
│ ├── 3_train_cnn.ipynb # Train CNN
│ └── 4_model_comparison.ipynb # Compare all models on test set ✅
├── README.md


---

## ✅ Models Overview

### 🔹 1. SVM + HOG
- Classic computer vision technique
- Uses OpenCV's HOG descriptor for feature extraction
- SVM classifier trained on HOG features

### 🔹 2. ANN (MLP)
- Input: Flattened grayscale image (128x64 → 8192 features)
- Built using Keras Sequential API with dense layers and dropout
- Fast and lightweight, but lower accuracy than CNN

### 🔹 3. CNN
- Input: 128×64 grayscale images with shape (128, 64, 1)
- Convolutional layers + MaxPooling + Dense layers
- Most accurate and generalizable model

---

## 📊 Model Evaluation (on same test set)

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| **SVM** | 95.51% | 95.11% | 92.55% | 93.81% |
| **ANN** | 83.40% | 77.78% | 76.81% | 77.29% |
| **CNN** | 98.32% | 98.73% | 96.69% | 97.70% |

All evaluations were performed using a common `X_test` and `y_test` split to ensure fair comparison.

---

## 📈 Visual Analysis

The comparison notebook includes:
- Classification reports
- Confusion matrices (SVM, ANN, CNN)
- ROC curve plot
- Visual predictions from all three models
- Summary table with performance metrics

📍 Notebook: `notebooks/4_model_comparison.ipynb`

---

## 🧪 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/human-detection-hog-mlp-cnn.git
cd human-detection-hog-mlp-cnn
2. Install dependencies
bash
Copy
Edit
pip install -r requirements.txt
3. Prepare the dataset
Place your images in the following folders:

bash
Copy
Edit
data/
├── pos_person/         # Human images
├── neg_person/         # Non-human images
4. Run Notebooks
Train each model using:

1_train_svm.ipynb

2_train_ann.ipynb

3_train_cnn.ipynb

Compare models using:

4_model_comparison.ipynb

📚 Tech Stack
Python 3.10+

OpenCV

scikit-learn

TensorFlow / Keras

Matplotlib, Seaborn, Pandas

Jupyter Notebook

✍️ Author
👤 Hanumat Lal Vishwakarma
📎 LinkedIn
📎 GitHub
