import os
import cv2
import numpy as np
import pandas as pd


def extract_features(image_path):
    print("Processing image:", image_path)
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Unable to read image at path:", image_path)
        return None
    avg_intensity = np.mean(img)
    return avg_intensity

infected_dir = "newdata/infected/"
uninfected_dir = "newdata/uninfected/"

features = []
labels = []


for image_name in os.listdir(infected_dir):
    image_path = os.path.join(infected_dir, image_name)
    feature = extract_features(image_path)
    features.append(feature)
    labels.append(1)

for image_name in os.listdir(uninfected_dir):
    image_path = os.path.join(uninfected_dir, image_name)
    feature = extract_features(image_path)
    features.append(feature)
    labels.append(0)

data = pd.DataFrame({"Feature": features, "Label": labels})

data.to_csv("new_dataset.csv", index=False)
