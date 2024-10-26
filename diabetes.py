# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Load dataset
data = pd.read_csv('../RETINA BACKEND PYTHON APP/dataset/diabetes.csv')

# Separate features and target variable
X = data.drop(columns=['Outcome']).values
y = data['Outcome'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a sequential model
model = Sequential()

# Add layers to the model with dropout and batch normalization
model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks for early stopping and model checkpointing
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', mode='max', save_best_only=True)

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, 
                    validation_split=0.2, callbacks=[early_stopping, model_checkpoint])

# Re-create the model to load weights
loaded_model = Sequential()
loaded_model.add(Dense(128, activation='relu', input_shape=(X.shape[1],)))
loaded_model.add(BatchNormalization())
loaded_model.add(Dropout(0.2))

loaded_model.add(Dense(256, activation='relu'))
loaded_model.add(BatchNormalization())
loaded_model.add(Dropout(0.3))

loaded_model.add(Dense(128, activation='relu'))
loaded_model.add(BatchNormalization())
loaded_model.add(Dropout(0.3))

loaded_model.add(Dense(64, activation='relu'))
loaded_model.add(BatchNormalization())
loaded_model.add(Dropout(0.3))

loaded_model.add(Dense(1, activation='sigmoid'))

# Compile the model again
loaded_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load the best weights from the training phase
loaded_model.load_weights('best_model.keras')

# Predict the classes on test data
y_pred = (loaded_model.predict(X_test) > 0.5).astype("int32")

# Evaluate the model on test data
test_loss, test_accuracy = loaded_model.evaluate(X_test, y_test)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Generate confusion matrix and classification report
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, loaded_model.predict(X_test))
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# Plot training and validation accuracies
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracies')
plt.legend()
plt.show()
