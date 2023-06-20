import cv2
import os
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skimage.feature import hog

# Step 1: Data Collection
# Assume you have collected and labeled images in the 'dataset' folder.
# Images with people are labeled as '1', and images without people are labeled as '0'.
# Load dataset and labels from files or directories
dataset_path = "path_to_dataset_folder"

dataset = []
labels = []

# Iterate through the dataset folder
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        if file.endswith(".jpg") or file.endswith(".png"):
            image_path = os.path.join(root, file)
            image = cv2.imread(image_path)
            dataset.append(image)
            
            # Extract the label from the folder name
            label = os.path.basename(root)
            labels.append(label)

# Print the number of images collected
print("Number of images:", len(dataset))


# Step 2: Preprocessing and Feature Extraction
# Convert images to grayscale and extract HOG features

def extract_hog_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    return features

hog_features = []

for image in dataset:
    features = extract_hog_features(image)
    hog_features.append(features)

# Step 3: Training
# Split the dataset into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(hog_features, labels, test_size=0.2, random_state=42)

# Create and train the SVM classifier

clf = svm.SVC()
clf.fit(X_train, y_train)

# Step 4: Model Evaluation
# Evaluate the classifier on the testing set

y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 5: Integration and Real-time Detection
# OpenCV integration for real-time detection from a webcam feed

def detect_people(frame):
    features = extract_hog_features(frame)
    prediction = clf.predict([features])
    return prediction

cap = cv2.VideoCapture(0)  # Use the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame if needed
    # frame = cv2.resize(frame, (640, 480))

    prediction = detect_people(frame)
    
    if prediction == 1:
        cv2.putText(frame, "Person Detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Human Detection", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
