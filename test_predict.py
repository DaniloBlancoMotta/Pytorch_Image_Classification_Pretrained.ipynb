#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unit tests for the prediction API
Run with: pytest test_predict.py -v
"""

import pytest
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path is set
from predict import app

@pytest.fixture
def client():
    """Create test client"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_endpoint(client):
    """Test home endpoint returns service info"""
    response = client.get('/')
    assert response.status_code == 200
    data = response.get_json()
    assert 'service' in data
    assert data['service'] == 'Bean Leaf Disease Classifier'
    assert 'version' in data
    assert 'endpoints' in data

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/health')
    assert response.status_code == 200
    data = response.get_json()
    assert 'status' in data
    assert data['status'] in ['healthy', 'unhealthy']
    assert 'model_loaded' in data
    assert 'device' in data

def test_info_endpoint(client):
    """Test model info endpoint"""
    response = client.get('/info')
    # Can be 200 or 500 depending on model availability
    assert response.status_code in [200, 500]
    
    if response.status_code == 200:
        data = response.get_json()
        assert 'model_type' in data
        assert 'num_classes' in data
        assert 'classes' in data

def test_predict_endpoint_no_file(client):
    """Test predict endpoint without file"""
    response = client.post('/predict')
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data

def test_predict_endpoint_empty_filename(client):
    """Test predict endpoint with empty filename"""
    data = {'file': (None, '')}
    response = client.post('/predict', data=data, content_type='multipart/form-data')
    assert response.status_code == 400

def test_predict_base64_endpoint_no_data(client):
    """Test predict_base64 endpoint without data"""
    response = client.post('/predict_base64', json={})
    assert response.status_code == 400
    data = response.get_json()
    assert 'error' in data

def test_predict_base64_endpoint_invalid_json(client):
    """Test predict_base64 endpoint with invalid JSON"""
    response = client.post('/predict_base64', data='invalid')
    assert response.status_code in [400, 415]  # Bad Request or Unsupported Media Type

def test_invalid_endpoint(client):
    """Test invalid endpoint returns 404"""
    response = client.get('/invalid_endpoint')
    assert response.status_code == 404

def test_health_endpoint_returns_json(client):
    """Test health endpoint returns valid JSON"""
    response = client.get('/health')
    assert response.content_type == 'application/json'

def test_home_endpoint_structure(client):
    """Test home endpoint JSON structure"""
    response = client.get('/')
    data = response.get_json()
    
    # Check required keys
    required_keys = ['service', 'version', 'status', 'endpoints', 'classes']
    for key in required_keys:
        assert key in data, f"Missing required key: {key}"

def test_endpoints_documented(client):
    """Test that all endpoints are documented in home"""
    response = client.get('/')
    data = response.get_json()
    endpoints = data.get('endpoints', {})
    
    expected_endpoints = ['/health', '/predict', '/predict_base64']
    for endpoint in expected_endpoints:
        assert endpoint in endpoints, f"Endpoint {endpoint} not documented"

class TestClassMapping:
    """Test class name mapping"""
    
    def test_class_ids(self, client):
        """Test that class IDs are consistent"""
        response = client.get('/')
        data = response.get_json()
        classes = data.get('classes', {})
        
        # Should have 3 classes (0, 1, 2)
        assert len(classes) >= 3
        assert '0' in classes or 0 in classes
        
    def test_class_names(self, client):
        """Test that class names are meaningful"""
        response = client.get('/')
        data = response.get_json()
        classes = data.get('classes', {})
        
        # Check that values are strings (class names)
        for class_id, class_name in classes.items():
            assert isinstance(class_name, str)
            assert len(class_name) > 0

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--cov=predict', '--cov-report=html'])
