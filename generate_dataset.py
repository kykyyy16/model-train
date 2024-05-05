import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def extract_features(image_path):
    print("Processing image:", image_path)
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to read image at path:", image_path)
        return None
    avg_intensity = np.mean(img)
    return avg_intensity

infected_dir = "dataset/Infected/"
uninfected_dir = "dataset/Uninfected/"

features = []
labels = []

# Extract features and labels from infected images
for image_name in os.listdir(infected_dir):
    image_path = os.path.join(infected_dir, image_name)
    feature = extract_features(image_path)
    if feature is not None:
        features.append(feature)
        labels.append(1)

# Extract features and labels from uninfected images
for image_name in os.listdir(uninfected_dir):
    image_path = os.path.join(uninfected_dir, image_name)
    feature = extract_features(image_path)
    if feature is not None:
        features.append(feature)
        labels.append(0)

# Create DataFrame from features and labels
data = pd.DataFrame({"Feature": features, "Label": labels})

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data["Feature"], data["Label"], test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(np.array(X_train).reshape(-1, 1), y_train)

# Predict labels for validation set
y_pred = model.predict(np.array(X_val).reshape(-1, 1))

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print("Validation Accuracy:", accuracy)
