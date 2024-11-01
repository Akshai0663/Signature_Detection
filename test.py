from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load your trained Keras model
model = load_model('signature_classification_model.h5')

# Mapping of labels
label_map = {
    0: 'Akshara',
    1: 'Akshai_Asok',
    2: 'Chacochan'
}

# Create an uploads folder if it doesn't exist
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    # Save the uploaded file to the uploads folder
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Read the image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))  # Resize to the model's input size
    image = img_to_array(image) / 255.0  # Normalize the image
    image = np.reshape(image, (1, 128, 128, 1))  # Reshape for the model

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_label = label_map[predicted_class]

    return render_template('result.html', filename=file.filename, predicted_label=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)