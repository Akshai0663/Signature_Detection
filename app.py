import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Dataset directory
DATASET_DIR = 'dataset/'

# Load dataset
data = []
labels = []

# Mapping of person to labels
label_map = {
    'Akshara': 0,
    'Akshai_Asok': 1,
    'Chacochan': 2
}

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    if os.path.isdir(person_path):
        label = label_map[person]
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (128, 128))  # Resize to the model's input size
                data.append(image)
                labels.append(label)

# Convert to NumPy arrays
data = np.array(data)
labels = np.array(labels)

# Normalize data
data = data / 255.0
data = np.reshape(data, (data.shape[0], 128, 128, 1))

# One-hot encode labels
labels = to_categorical(labels, num_classes=3)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=16)

# Save the model
model.save('signature_classification_model.h5')
