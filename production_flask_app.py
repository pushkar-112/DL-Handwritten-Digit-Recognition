# app_production.py - Clean Production Flask Application
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import cv2
import os

app = Flask(__name__)

# Global variable to store the model
model = None

def load_model():
    """Load the trained digit classifier model"""
    global model
    try:
        
        model_loaded = False
         # Make path platform-independent
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, "model", "digit_recognition_model.h5")

        if os.path.exists(model_path):
            try:
                model = tf.keras.models.load_model(model_path)
                print(f"Model loaded successfully from: {model_path}")
                print(f"  Input shape: {model.input_shape}")
                print(f"  Output shape: {model.output_shape}")
                model_loaded = True
                    
            except Exception as load_error:
                print(f"Error loading model from {model_path}: {load_error}")
                  
        
        if not model_loaded:
            print("Model not found. Please ensure your .h5 file is in the correct location.")
                
    except Exception as e:
        print(f"Critical error in load_model: {e}")

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert PIL image to numpy array
        img_array = np.array(image)
        
        # Convert to grayscale if it's RGB
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to 28x28 (MNIST standard)
        img_array = cv2.resize(img_array, (28, 28))
        
        # Handle different image types and backgrounds
        # Check if the image has a light or dark background
        mean_value = np.mean(img_array)
        
        # If the background is light (mean > 127), invert the image
        # This makes it consistent with MNIST (dark digits on light background)
        if mean_value > 127:
            img_array = 255 - img_array  # Invert
        
        # Apply enhancement to make digits more prominent
        # Increase contrast
        img_array = cv2.convertScaleAbs(img_array, alpha=1.2, beta=10)
        
        # Apply threshold to make it more binary (like MNIST)
        _, img_array = cv2.threshold(img_array, 128, 255, cv2.THRESH_BINARY)
        
        # Normalize pixel values to 0-1 range
        img_array = img_array.astype('float32') / 255.0
        
        # Flatten the image to match model's expected input (784,)
        img_array = img_array.reshape(1, 784)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def preprocess_canvas_data(canvas_data):
    """Preprocess canvas drawing data"""
    try:
        # Remove the data URL prefix
        image_data = canvas_data.split(',')[1]
        
        # Decode base64 image
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to grayscale
        image = image.convert('L')
        
        # Convert to numpy array and invert colors (white background -> black, black drawing -> white)
        img_array = np.array(image)
        img_array = 255 - img_array  # Invert
        
        # Resize to 28x28
        img_array = cv2.resize(img_array, (28, 28))
        
        # Normalize
        img_array = img_array.astype('float32') / 255.0
        
        # Flatten the image to match model's expected input (784,)
        img_array = img_array.reshape(1, 784)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing canvas data: {e}")
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index_production.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please check if the model file exists.'
            }), 500
        
        # Check if it's an uploaded file or canvas data
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Process uploaded image
            image = Image.open(file.stream)
            processed_image = preprocess_image(image)
            
        elif 'canvas_data' in request.json:
            # Process canvas drawing
            canvas_data = request.json['canvas_data']
            processed_image = preprocess_canvas_data(canvas_data)
            
        else:
            return jsonify({'error': 'No image data provided'}), 400
        
        if processed_image is None:
            return jsonify({'error': 'Failed to process image'}), 400
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get all probabilities
        probabilities = [float(prob) for prob in predictions[0]]
        
        return jsonify({
            'predicted_digit': int(predicted_class),
            'confidence': confidence,
            'probabilities': probabilities
        })
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Load the model when starting the app
    load_model()
    
    # Run the Flask app
    app.run(debug=False, host='0.0.0.0', port=5000)