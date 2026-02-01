#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prediction Service for PyTorch Image Classification
Serves trained model via Flask API
"""

import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision import models
from flask import Flask, request, jsonify
from PIL import Image
import pickle
import io
import base64
import numpy as np

app = Flask('bean-leaf-classifier')

# Global variables for model and preprocessing
model = None
label_encoder = None
device = None
transform = None

def load_model(model_path='model.pkl'):
    """Load trained model and preprocessing objects"""
    global model, label_encoder, device, transform
    
    print("Loading model...")
    
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Extract components
        label_encoder = model_data['label_encoder']
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Recreate model architecture
        num_classes = len(label_encoder.classes_)
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
        
        # Load weights
        model.load_state_dict(model_data['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Define transforms
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Model loaded successfully on {device}")
        print(f"Classes: {label_encoder.classes_}")
        
    except FileNotFoundError:
        print(f"Error: Model file '{model_path}' not found!")
        print("Please train the model first using: python train.py")
        raise
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def preprocess_image(image):
    """Preprocess image for model input"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    image_tensor = transform(image)
    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor.to(device)

def predict_image(image):
    """Make prediction on image"""
    # Preprocess
    image_tensor = preprocess_image(image)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get class name
    predicted_class = label_encoder.inverse_transform([predicted.item()])[0]
    confidence_score = confidence.item()
    
    # Get all class probabilities
    all_probabilities = {
        label_encoder.inverse_transform([i])[0]: prob.item() 
        for i, prob in enumerate(probabilities[0])
    }
    
    return {
        'prediction': int(predicted_class),
        'class_name': get_class_name(predicted_class),
        'confidence': float(confidence_score),
        'probabilities': all_probabilities
    }

def get_class_name(class_id):
    """Convert class ID to human-readable name"""
    class_names = {
        0: 'Angular Leaf Spot',
        1: 'Bean Rust',
        2: 'Healthy'
    }
    return class_names.get(class_id, f'Class {class_id}')

@app.route('/')
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        'service': 'Bean Leaf Disease Classifier',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            '/health': 'Health check',
            '/predict': 'POST - Image prediction (upload file)',
            '/predict_base64': 'POST - Image prediction (base64 encoded)'
        },
        'classes': {
            0: 'Angular Leaf Spot',
            1: 'Bean Rust',
            2: 'Healthy'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint for uploaded images"""
    try:
        # Check if image file is in request
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file provided',
                'message': 'Please upload an image file'
            }), 400
        
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            return jsonify({
                'error': 'Empty filename',
                'message': 'Please select a file'
            }), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        result = predict_image(image)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Prediction endpoint for base64 encoded images"""
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({
                'error': 'No image data provided',
                'message': 'Please provide base64 encoded image in JSON: {"image": "base64_string"}'
            }), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Make prediction
        result = predict_image(image)
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/info', methods=['GET'])
def info():
    """Model information endpoint"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'ResNet18 (Pretrained)',
        'num_classes': len(label_encoder.classes_),
        'classes': {
            int(i): {'id': int(i), 'name': get_class_name(i)} 
            for i in label_encoder.classes_
        },
        'input_size': '128x128',
        'device': str(device)
    })

def create_test_request_example():
    """Create example for testing the API"""
    print("\n" + "="*50)
    print("API Test Examples")
    print("="*50)
    
    print("\n1. Test with file upload (curl):")
    print("""
    curl -X POST http://localhost:9696/predict \\
      -F "file=@path/to/image.jpg"
    """)
    
    print("\n2. Test with Python:")
    print("""
    import requests
    
    # Upload file
    with open('image.jpg', 'rb') as f:
        files = {'file': f}
        response = requests.post('http://localhost:9696/predict', files=files)
    
    print(response.json())
    """)
    
    print("\n3. Check health:")
    print("    curl http://localhost:9696/health")
    
    print("\n" + "="*50 + "\n")

if __name__ == '__main__':
    # Load model at startup
    load_model('model.pkl')
    
    # Print test examples
    create_test_request_example()
    
    # Start Flask app
    print("Starting Flask service on http://0.0.0.0:9696")
    app.run(debug=True, host='0.0.0.0', port=9696)
