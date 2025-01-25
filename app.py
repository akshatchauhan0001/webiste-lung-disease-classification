import os
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from numpy.linalg import norm

# Initialize Flask app
app = Flask(__name__)

# Load the trained fully connected model
model = load_model('final_model_resnet50_and_fully_connected_nn.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

# Load ResNet50 model for feature extraction (without the top classification layer)
resnet_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# Define class names
class_name = {0: 'COVID', 1: 'Lung Opacity', 2: 'Normal', 3: 'Pneumonia'}



UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload():
    global file
    global file_path
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return render_template('second.html', image_file=file.filename)


def extract_features(image_path):
    """
    Extract features from an image using the ResNet50 model.
    """
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x) # Preprocess the image for ResNet50
    features = resnet_model.predict(x)
    return features


def getResult(image_path):
    """
    Predict the class of the given image using the trained model.
    """
    # Extract features using ResNet50
    features = extract_features(image_path)

    # Normalize the features
    features = features / norm(features)

    # Predict the class using the trained fully connected model
    predictions = model.predict(features)[0]
    return predictions


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/second')
def second():
    return render_template('second.html')

@app.route('/covid_sample')
def covid_sample():
    return render_template('covid_sample.html')

@app.route('/pneumonia_sample')
def pneumonia_sample():
    return render_template('pneumonia_sample.html')

@app.route('/opacity_sample')
def opacity_sample():
    return render_template('opacity_sample.html')

@app.route('/normal_sample')
def normal_sample():
    return render_template('normal_sample.html')

@app.route('/back')
def back():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # if 'file' not in request.files:
    #     return redirect(request.url)
    #
    # file = request.files['file']
    #
    # if file.filename == '':
    #     return redirect(request.url)
    #
    # # Save the uploaded file
    # file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    # file.save(file_path)

    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)


    # Get predictions
        predictions = getResult(file_path)
        predicted_label = class_name[np.argmax(predictions)]
        confidence = round(100 * np.max(predictions), 2)

    # Construct prediction result
        prediction_result = f"Predicted Class : {predicted_label}  |  Confidence: {confidence}%"
        return render_template('second.html', prediction=prediction_result,image_file=file.filename)

# Import the LIME explanation function
from lime_explanation import explain_and_save_lime_image  # Assuming the above function is in lime_explanation.py

UPLOAD_FOLDER = 'static/uploads/'
OUTPUT_FOLDER = 'static/LIME_OUTPUTS/'  # Updated folder for LIME outputs
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/four', methods=['GET', 'POST'])
def four():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400

        img_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(img_path)

        # Generate LIME explanation and save in LIME_OUTPUTS
        output_image = explain_and_save_lime_image(img_path, OUTPUT_FOLDER)

        # Get relative path for displaying in HTML (inside LIME_OUTPUTS folder)
        relative_output_image = os.path.relpath(output_image, 'static')  # Correcting relative path
        return render_template('four.html', lime_image=relative_output_image)

    return render_template('four.html', lime_image=None)

if __name__ == '__main__':
    app.run(debug=True)