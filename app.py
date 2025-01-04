import os
import gdown
import zipfile
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import numpy as np
import cv2
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Function to download and extract datasets/models from Google Drive
def download_and_extract():
    # URLs for Google Drive files (replace with your actual file IDs)
    dataset_url = 'https://drive.google.com/uc?export=download&id=1CDlprA0zbj9wXTRuG2Z0PdnQrhwjgtrY'  # Dataset ZIP
    model_url = 'https://drive.google.com/uc?export=download&id=1Kk2FMy7Enj6zjK66RnPDryRjBiVHUC5x'  # Models ZIP

    # Temporary file names
    dataset_zip = 'dataset.zip'
    model_zip = 'models.zip'
    dataset_folder = 'dataset_folder'
    model_folder = 'models'

    # Download the Dataset ZIP file
    if not os.path.exists(dataset_zip):
        print("Downloading dataset ZIP file...")
        gdown.download(dataset_url, dataset_zip, quiet=False)

    # Download the Models ZIP file
    if not os.path.exists(model_zip):
        print("Downloading models ZIP file...")
        gdown.download(model_url, model_zip, quiet=False)

    # Extract the dataset if not already done
    if not os.path.exists(dataset_folder):
        print("Extracting dataset ZIP file...")
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(dataset_folder)
        print(f"Dataset extracted to {dataset_folder}")

    # Extract the models if not already done
    if not os.path.exists(model_folder):
        print("Extracting models ZIP file...")
        with zipfile.ZipFile(model_zip, 'r') as zip_ref:
            zip_ref.extractall(model_folder)
        print(f"Models extracted to {model_folder}")

    # Clean up ZIP files
    if os.path.exists(dataset_zip):
        os.remove(dataset_zip)
    if os.path.exists(model_zip):
        os.remove(model_zip)

# Call the function to download and extract datasets/models
download_and_extract()

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'  # Directory for saving uploads
app.secret_key = 'supersecretkey'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load Models
numerical_model = tf.keras.models.load_model("breast_cancer_model.h5")
image_classification_model = tf.keras.models.load_model("segmentation_model_classification.h5")
segmentation_model = tf.keras.models.load_model("segmentation_model.h5")

# Load Dataset for Scaling Numerical Features
data = pd.read_csv("dataset_folder/data.csv")
data_cleaned = data.drop(columns=['id', 'Unnamed: 32'])
X = data_cleaned.drop(columns=['diagnosis'])
scaler = StandardScaler()
scaler.fit(X)

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
    input_image, original_image = process_image(image_path)

    # Classification
    class_prediction = image_classification_model.predict(input_image)
    predicted_class = np.argmax(class_prediction)
    class_names = ["Benign", "Malignant", "Normal"]
    predicted_class_name = class_names[predicted_class]

    # Segmentation
    segmentation_prediction = segmentation_model.predict(input_image)[0]
    segmented_mask = (segmentation_prediction > 0.5).astype(np.uint8)

    # Overlay segmentation mask
    overlay = cv2.addWeighted(original_image, 0.7, cv2.cvtColor(segmented_mask * 255, cv2.COLOR_GRAY2BGR), 0.3, 0)

    # Save outputs
    overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], 'overlay_image.png')
    mask_path = os.path.join(app.config['UPLOAD_FOLDER'], 'segmentation_mask.png')
    cv2.imwrite(overlay_path, overlay)
    cv2.imwrite(mask_path, segmented_mask * 255)

    return predicted_class_name, overlay_path, mask_path

def classify_features(features):
    """Classify using numerical features."""
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
        return "Error: No input provided. Please upload an image or enter feature values."

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
