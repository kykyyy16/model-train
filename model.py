import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Load data
data = pd.read_csv('new_dataset.csv')

# Drop rows with missing values
data.dropna(inplace=True)

print(data.columns)
print(data.drop(columns=['Label']))

# Pairplot
sns.pairplot(data.drop(columns=['Label']))  # Drop the Label column if it's present in the dataset
plt.show()

# Class distribution
sns.countplot(x='Label', data=data)
plt.title('Class Distribution')
plt.show()

# Define features (X) and Label (y)
X = data.drop(columns=['Label']).values
y = data['Label'].values

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Data Augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20, 
    width_shift_range=0.1, 
    height_shift_range=0.1,  
    shear_range=0.2, 
    zoom_range=0.2, 
    horizontal_flip=True, 
    vertical_flip=False) 

# Train CNN model
cnn_model = tf.keras.Sequential([
  tf.keras.layers.Input(shape=(X_train.shape[1],)),
  tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Implement early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = cnn_model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Evaluate CNN model
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, y_test)
print("\nCNN Accuracy:", cnn_accuracy)

cnn_pred = (cnn_model.predict(X_test) > 0.5).astype("int32")
cnn_cm = confusion_matrix(y_test, cnn_pred)
cnn_report = classification_report(y_test, cnn_pred, output_dict=True)

print("\nCNN Confusion Matrix:")
print(cnn_cm)
print("\nCNN Classification Report:")
print(cnn_report)

# Train SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_report = classification_report(y_test, svm_pred, output_dict=True)
svm_cm = confusion_matrix(y_test, svm_pred)

print("\nSVM Accuracy:", svm_accuracy)
print("\nSVM Confusion Matrix:")
print(svm_cm)
print("\nSVM Classification Report:")
print(svm_report)

# Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=5, min_samples_leaf=2, max_features='sqrt', random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_report = classification_report(y_test, rf_pred, output_dict=True)
rf_cm = confusion_matrix(y_test, rf_pred)

print("\nRandom Forest Accuracy:", rf_accuracy)
print("\nRandom Forest Confusion Matrix:")
print(rf_cm)
print("\nRandom Forest Classification Report:")
print(rf_report)


# Fusion of models using stacking with weighted combination

# Fusion of SVM, CNN, RF using stacking
base_model_predictions = np.column_stack((svm_pred, cnn_pred, rf_pred))

# Initialize meta-model (Logistic Regression)
meta_model = LogisticRegression()

# Use StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

stacked_accuracies = []

for train_index, val_index in skf.split(base_model_predictions, y_test):
    meta_model.fit(base_model_predictions[train_index], y_test[train_index])

    stacked_pred = meta_model.predict(base_model_predictions[val_index])

    stacked_accuracy = accuracy_score(y_test[val_index], stacked_pred)
    stacked_accuracies.append(stacked_accuracy)

# Predict using the trained meta-model
stacked_pred = meta_model.predict(base_model_predictions)

average_stacked_accuracy = np.mean(stacked_accuracies)

print("\nStacked Fusion Confusion Matrix:")
print(confusion_matrix(y_test, stacked_pred))
print("\nStacked Fusion Classification Report:")
print(classification_report(y_test, stacked_pred, target_names=['uninfected', 'infected']))
print("\nStacked Fusion Accuracy:", average_stacked_accuracy)

# Fusion of SVM and CNN using stacking
base_model_predictions_svm_cnn = np.column_stack((svm_pred, cnn_pred))

# Initialize meta-model (Logistic Regression)
meta_model_svm_cnn = LogisticRegression()

# StratifiedKFold for cross-validation
skf_svm_cnn = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in skf_svm_cnn.split(base_model_predictions_svm_cnn, y_test):
    meta_model_svm_cnn.fit(base_model_predictions_svm_cnn[train_index], y_test[train_index])

stacked_pred_svm_cnn = meta_model_svm_cnn.predict(base_model_predictions_svm_cnn)

stacked_accuracy_svm_cnn = accuracy_score(y_test, stacked_pred_svm_cnn)

print("\nStacked Fusion (SVM + CNN) Confusion Matrix:")
print(confusion_matrix(y_test, stacked_pred_svm_cnn))
print("\nStacked Fusion (SVM + CNN) Classification Report:")
print(classification_report(y_test, stacked_pred_svm_cnn, target_names=['uninfected', 'infected']))
print("Stacked Fusion (SVM + CNN) Accuracy:", stacked_accuracy_svm_cnn)

base_model_predictions_svm_rf = np.column_stack((svm_pred, rf_pred))

meta_model_svm_rf = LogisticRegression()

skf_svm_rf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in skf_svm_rf.split(base_model_predictions_svm_rf, y_test):
    meta_model_svm_rf.fit(base_model_predictions_svm_rf[train_index], y_test[train_index])

stacked_pred_svm_rf = meta_model_svm_rf.predict(base_model_predictions_svm_rf)

stacked_accuracy_svm_rf = accuracy_score(y_test, stacked_pred_svm_rf)

print("\nStacked Fusion (SVM + RF) Confusion Matrix:")
print(confusion_matrix(y_test, stacked_pred_svm_rf))
print("\nStacked Fusion (SVM + RF) Classification Report:")
print(classification_report(y_test, stacked_pred_svm_rf, target_names=['uninfected', 'infected']))
print("Stacked Fusion (SVM + RF) Accuracy:", stacked_accuracy_svm_rf)

base_model_predictions_cnn_rf = np.column_stack((cnn_pred, rf_pred))

meta_model_cnn_rf = LogisticRegression()

skf_cnn_rf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_index, val_index in skf_cnn_rf.split(base_model_predictions_cnn_rf, y_test):
    meta_model_cnn_rf.fit(base_model_predictions_cnn_rf[train_index], y_test[train_index])

stacked_pred_cnn_rf = meta_model_cnn_rf.predict(base_model_predictions_cnn_rf)

stacked_accuracy_cnn_rf = accuracy_score(y_test, stacked_pred_cnn_rf)

print("\nStacked Fusion (CNN + RF) Confusion Matrix:")
print(confusion_matrix(y_test, stacked_pred_cnn_rf))
print("\nStacked Fusion (CNN + RF) Classification Report:")
print(classification_report(y_test, stacked_pred_cnn_rf, target_names=['uninfected', 'infected']))
print("Stacked Fusion (CNN + RF) Accuracy:", stacked_accuracy_cnn_rf)

def plot_confusion_matrix(cm, labels, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(svm_cm, ['uninfected', 'infected'], 'SVM')
plot_confusion_matrix(cnn_cm, ['uninfected', 'infected'], 'CNN')
plot_confusion_matrix(rf_cm, ['uninfected', 'infected'], 'RF')
plot_confusion_matrix(confusion_matrix(y_test, stacked_pred), ['uninfected', 'infected'], 'Stacked Fusion')
plot_confusion_matrix(confusion_matrix(y_test, stacked_pred_svm_cnn), ['uninfected', 'infected'], 'Stacked Fusion (SVM + CNN)')
plot_confusion_matrix(confusion_matrix(y_test, stacked_pred_svm_rf), ['uninfected', 'infected'], 'Stacked Fusion (SVM + RF)')
plot_confusion_matrix(confusion_matrix(y_test, stacked_pred_cnn_rf), ['uninfected', 'infected'], 'Stacked Fusion (CNN + RF)')

labels = ['SVM+CNN', 'SVM+RF', 'CNN+RF', 'SVM+CNN+RF']
fusion_accuracies = [stacked_accuracy_svm_cnn, stacked_accuracy_svm_rf, stacked_accuracy_cnn_rf, stacked_accuracy]

# Calculate base model accuracies
base_accuracies = [svm_accuracy, cnn_accuracy, rf_accuracy]

# Calculate fusion model accuracies
fusion_accuracies = [stacked_accuracy_svm_cnn, stacked_accuracy_svm_rf, stacked_accuracy_cnn_rf, stacked_accuracy]


# Plot CNN regression
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Model Accuracy')
plt.legend()
plt.show()

# Plot accuracy per base class
plt.figure(figsize=(10, 6))
plt.bar(['SVM', 'CNN', 'RF'], base_accuracies, color='blue')
plt.xlabel('Base Models')
plt.ylabel('Accuracy')
plt.title('Accuracy per Base Class')
plt.show()

# Plot accuracy of fusion models
plt.figure(figsize=(6, 4))
plt.bar(['Stacked Fusion'], fusion_accuracies, color='green')
plt.xlabel('Fusion Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Fusion Models')
plt.show()

# Plot accuracy of base models and fusion of SVM, CNN, and RFs
labels = ['SVM', 'CNN', 'RF', 'Stacked Fusion (SVM + CNN + RF)']
accuracies = [svm_accuracy, cnn_accuracy, rf_accuracy, stacked_accuracy]

# Plot the bar graph
plt.figure(figsize=(10, 6))
plt.bar(labels, accuracies, color=['blue', 'orange', 'green', 'purple'])
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Base Models and Stacked Fusion Model')
plt.show()