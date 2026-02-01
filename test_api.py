#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for the prediction API
"""

import requests
import base64
import sys
from pathlib import Path

API_URL = "http://localhost:9696"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*50)
    print("Testing /health endpoint")
    print("="*50)
    
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_info():
    """Test info endpoint"""
    print("\n" + "="*50)
    print("Testing /info endpoint")
    print("="*50)
    
    try:
        response = requests.get(f"{API_URL}/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_file(image_path):
    """Test predict endpoint with file upload"""
    print("\n" + "="*50)
    print("Testing /predict endpoint with file upload")
    print("="*50)
    
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{API_URL}/predict", files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_base64(image_path):
    """Test predict endpoint with base64 encoded image"""
    print("\n" + "="*50)
    print("Testing /predict_base64 endpoint")
    print("="*50)
    
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return False
    
    try:
        # Read and encode image
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Send request
        data = {'image': image_base64}
        response = requests.post(f"{API_URL}/predict_base64", json=data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("API Testing Script")
    print("="*70)
    print(f"API URL: {API_URL}")
    
    # Test health
    health_ok = test_health()
    
    # Test info
    info_ok = test_info()
    
    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("\nNo image path provided. Skipping prediction tests.")
        print("Usage: python test_api.py path/to/image.jpg")
        image_path = None
    
    # Test prediction endpoints
    predict_file_ok = False
    predict_base64_ok = False
    
    if image_path:
        predict_file_ok = test_predict_file(image_path)
        predict_base64_ok = test_predict_base64(image_path)
    
    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)
    print(f"✓ Health endpoint: {'PASS' if health_ok else 'FAIL'}")
    print(f"✓ Info endpoint: {'PASS' if info_ok else 'FAIL'}")
    
    if image_path:
        print(f"✓ Predict (file) endpoint: {'PASS' if predict_file_ok else 'FAIL'}")
        print(f"✓ Predict (base64) endpoint: {'PASS' if predict_base64_ok else 'FAIL'}")
    else:
        print("⊘ Prediction tests skipped (no image provided)")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
