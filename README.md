-----

# 🚶 Human Detection System 🚶‍♀️

## An OpenCV and Scikit-learn Powered Solution for Identifying Humans in Images

[](https://www.python.org/)
[](https://opencv.org/)
[](https://scikit-learn.org/stable/)
[](https://www.google.com/search?q=LICENSE) \#\# ✨ Overview

Welcome to the Human Detection System\! This project leverages the power of computer vision and machine learning to accurately identify human figures within static images. Built with **OpenCV** for image processing and **scikit-learn** for classification, this system employs the Histogram of Oriented Gradients (HOG) features combined with a Linear Support Vector Machine (SVM) to achieve robust human presence detection.

Whether you're building surveillance applications, robotics systems, or simply exploring the fascinating world of object detection, this repository provides a clear, concise, and effective starting point.

## 🚀 Features

  * **HOG Feature Extraction:** Utilizes the highly effective Histogram of Oriented Gradients (HOG) descriptor for robust feature representation of human shapes.
  * **Linear SVM Classifier:** Employs a Linear Support Vector Machine for efficient and accurate classification based on extracted HOG features.
  * **Modular Codebase:** Clean and well-structured code for easy understanding and extension.
  * **Model Persistence:** Ability to save and load the trained SVM model for future use without retraining.
  * **Comprehensive Evaluation:** Includes classification report and confusion matrix to assess model performance.
  * **Flexible Testing:** Supports testing on single or multiple new images to visualize detection capabilities.

## 🛠️ Installation

To get this project up and running on your local machine, follow these simple steps.

### Prerequisites

  * Python 3.x

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/A-Human-Detection-System-Using-Machine-Learning.git
cd A-Human-Detection-System-Using-Machine-Learning
```

*(Replace `your-username` with your actual GitHub username and the repository name if it differs)*

### 2\. Install Dependencies

All required libraries can be installed using `pip`:

```bash
pip install numpy matplotlib scikit-learn opencv-python joblib
```

You might see messages about "Requirement already satisfied" for some packages; this is normal if you have them installed. The crucial packages like `opencv-python` will be downloaded and installed.

## 📁 Project Structure

```
.
├── data/
│   ├── pos_person/         # Directory containing positive samples (human images)
│   ├── neg_person/         # Directory containing negative samples (non-human images)
│   └── predictTestImages/  # Directory for new images to test detection
├── model/
│   └── svm_model.pkl       # Trained SVM model will be saved here
├── human_detection.ipynb   # Jupyter Notebook containing the full code and explanations
└── README.md               # This file
```

*(Note: Adjust `human_detection.ipynb` if your main script is a `.py` file or if the notebook name is different)*

## 💡 Usage

The core logic of the human detection system is demonstrated within the Jupyter Notebook.

### 1\. Prepare Your Dataset

Ensure you have a collection of positive (human) and negative (non-human) images.

  * Place your human images in the `data/pos_person` directory.
  * Place your non-human images in the `data/neg_person` directory.

The code is configured to look for these paths:

```python
pos_path = "../data/pos_person"
neg_path = "../data/neg_person"
```

### 2\. Run the Jupyter Notebook

Open the `human_detection.ipynb` file in a Jupyter environment (e.g., Jupyter Lab or Jupyter Notebook):

```bash
jupyter notebook
```

Execute the cells sequentially. The notebook will guide you through:

  * **Importing Libraries:** Essential tools for data manipulation, image processing, and machine learning.
  * **Viewing Sample Images:** A quick visualization of the dataset's content.
  * **Preparing Dataset (HOG Features):** Extraction of HOG features from images and labeling them.
  * **Train/Test Split:** Dividing the dataset for model training and evaluation.
  * **Training SVM Classifier:** Training the Linear SVM model on the HOG features.
  * **Evaluating the Model:** Displaying classification report and confusion matrix to assess performance.
  * **Saving and Loading the Model:** Demonstrating how to persist the trained model.
  * **Testing Detection:** Applying the trained model on new, unseen images to predict the presence of humans.

### 3\. Test on Your Own Images

To test the system on new images:

1.  Place your desired test images in the `data/predictTestImages` folder.
2.  Run the final sections of the Jupyter Notebook to see the real-time predictions.

## 📊 Model Performance

The trained Linear SVM model demonstrates strong performance in distinguishing between human and non-human subjects based on HOG features.

### Classification Report:

```
              precision    recall  f1-score   support

           0       0.96      0.97      0.97       830  (Not Human)
           1       0.95      0.93      0.94       483  (Human)

    accuracy                           0.96      1313

   macro avg       0.96      0.95      0.95      1313
weighted avg       0.96      0.96      0.96      1313
```

### Confusion Matrix:

```
[[807  23]  (True Negative: 807, False Positive: 23)
 [ 33 450]] (False Negative: 33, True Positive: 450)
```

These metrics indicate high accuracy, precision, and recall, showcasing the model's effectiveness in identifying human figures with minimal false positives and false negatives.

## 📈 Future Enhancements

  * **Real-time Video Processing:** Extend the system to process live video streams for dynamic human detection.
  * **Deep Learning Integration:** Explore more advanced models like Convolutional Neural Networks (CNNs) for potentially higher accuracy and robustness in complex scenarios.
  * **Object Tracking:** Implement tracking algorithms to follow detected humans across sequential frames.
  * **More Diverse Dataset:** Expand the training dataset with more varied poses, lighting conditions, backgrounds, and occlusions to enhance model generalization.
  * **Optimization:** Investigate performance optimizations, including potential GPU acceleration for faster inference.

## 🤝 Contributing

Contributions are welcome\! If you have ideas for improvements, bug fixes, or new features, please feel free to open an [issue](https://www.google.com/search?q=https://github.com/hlvcse/A-Human-Detection-System-Using-Machine-Learning/issues) or submit a [pull request](https://www.google.com/search?q=https://github.com/hlvcse/A-Human-Detection-System-Using-Machine-Learning/pulls).


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.
*(Remember to create a `LICENSE` file if you plan to share your project publicly.)*

-----

Made with ❤️ by [Hanumat Lal Vishwakarma](https://www.google.com/search?q=https://github.com/hlvcse)
