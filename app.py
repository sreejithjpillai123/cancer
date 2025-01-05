import os
import gdown
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Function to download and prepare datasets/models
def download_and_prepare():
    dataset_url = 'https://drive.google.com/uc?export=download&id=1CDlprA0zbj9wXTRuG2Z0PdnQrhwjgtrY'

    dataset_zip = 'dataset.zip'
    dataset_folder = 'dataset_folder'
    model_folder = 'models'

    # Download the dataset ZIP file
    if not os.path.exists(dataset_zip):
        print("Downloading dataset ZIP file...")
        gdown.download(dataset_url, dataset_zip, quiet=False)

    # Extract the dataset if not already done
    if not os.path.exists(dataset_folder):
        print("Extracting dataset ZIP file...")
        import zipfile
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_folder)
        print(f"Dataset extracted to {dataset_folder}")

    # Clean up the dataset ZIP file
    if os.path.exists(dataset_zip):
        os.remove(dataset_zip)

    # Ensure the model folder exists
    if not os.path.exists(model_folder):
        print(f"Error: Model folder '{model_folder}' does not exist.")
        return False
    return True

# Prepare datasets and models
if not download_and_prepare():
    print("Error: Required resources are missing.")
    exit(1)

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.secret_key = 'supersecretkey'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load Models with Exception Handling
try:
    numerical_model = tf.keras.models.load_model("models/breast_cancer_model.h5")
    print("Numerical model loaded successfully.")
except Exception as e:
    numerical_model = None
    print(f"Error loading numerical model: {e}")

try:
    image_classification_model = tf.keras.models.load_model("models/segmentation_model_classification.h5")
    print("Image classification model loaded successfully.")
except Exception as e:
    image_classification_model = None
    print(f"Error loading image classification model: {e}")

try:
    segmentation_model = tf.keras.models.load_model("models/segmentation_model.h5")
    print("Segmentation model loaded successfully.")
except Exception as e:
    segmentation_model = None
    print(f"Error loading segmentation model: {e}")

# Load Dataset for Scaling Numerical Features
data_file = "dataset_folder/data.csv"
if os.path.exists(data_file):
    data = pd.read_csv(data_file)
    data_cleaned = data.drop(columns=['id', 'Unnamed: 32'])
    X = data_cleaned.drop(columns=['diagnosis'])
    scaler = StandardScaler()
    scaler.fit(X)
else:
    print("Dataset CSV file not found.")
    scaler = None

def process_image(image_path, img_size=128):
    """Preprocess an image for prediction."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    resized_image = cv2.resize(image, (img_size, img_size))
    normalized_image = resized_image / 255.0
    expanded_image = np.expand_dims(normalized_image, axis=0)
    return expanded_image, resized_image

def classify_image(image_path):
    """Classify an image and perform segmentation."""
    if image_classification_model is None or segmentation_model is None:
        raise ValueError("Required models are not loaded.")

    input_image, original_image = process_image(image_path)

    class_prediction = image_classification_model.predict(input_image)
    predicted_class = np.argmax(class_prediction)
    class_names = ["Benign", "Malignant", "Normal"]
    predicted_class_name = class_names[predicted_class]

    segmentation_prediction = segmentation_model.predict(input_image)[0]
    segmented_mask = (segmentation_prediction > 0.5).astype(np.uint8)

    overlay = cv2.addWeighted(original_image, 0.7, cv2.cvtColor(segmented_mask * 255, cv2.COLOR_GRAY2BGR), 0.3, 0)

    overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], 'overlay_image.png')
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmentation_mask.png')
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(mask_path, segmented_mask * 255)

    return predicted_class_name, overlay_path, mask_path

def classify_features(features):
    """Classify using numerical features."""
    if numerical_model is None or scaler is None:
        raise ValueError("Numerical model or scaler is not loaded.")
    features_scaled = scaler.transform([features])
    predictions = numerical_model.predict(features_scaled)
    predicted_class = (predictions > 0.5).astype(int)
    class_names = ["Benign", "Malignant"]
    return class_names[predicted_class[0][0]]

@app.route('/')
def index():
    return render_template('index_combined.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files and request.files['image'].filename != '':
        file = request.files['image']
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(image_path)

        try:
            predicted_class, overlay_path, mask_path = classify_image(image_path)
            return render_template(
                'result_combined.html',
                prediction=predicted_class,
                image_filename=file.filename,
                overlay_filename=overlay_path.split('/')[-1],
                mask_filename=mask_path.split('/')[-1]
            )
        except Exception as e:
            return f"Error processing the image: {e}"

    elif 'features' in request.form and request.form['features'] != '':
        raw_input = request.form['features']
        try:
            features = [float(value.strip()) for value in raw_input.split(",")]
            if len(features) != X.shape[1]:
                return "Error: Provide 30 feature values separated by commas."
            predicted_class = classify_features(features)
            return render_template(
                'result_combined.html',
                prediction=predicted_class,
                image_filename=None,
                overlay_filename=None,
                mask_filename=None
            )
        except Exception as e:
            return f"Error processing values: {e}"
    else:
        return "Error: No input provided. Please upload an image file or enter 30 comma-separated feature values."

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
