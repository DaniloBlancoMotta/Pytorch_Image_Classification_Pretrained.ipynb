---
name: ML Project Structure & Deployment
description: Complete guide for creating production-ready machine learning projects with proper structure, documentation, and deployment capabilities
---

# ML Project Structure & Deployment Skill

## Overview
This skill guides the creation of complete, production-ready machine learning projects with proper structure, documentation, and deployment capabilities.

## When to Use This Skill
Use this skill when:
- Building an end-to-end ML project from scratch
- Creating a deployable ML service
- Structuring a data science portfolio project
- Need to follow ML engineering best practices
- Building projects for courses, bootcamps, or professional work

## Project Structure Requirements

### 1. README.md
**Purpose**: Project documentation and entry point for users

**Required Sections**:
- **Problem Description**: Clear explanation of the business/technical problem being solved
- **Dataset Description**: What data is used, features, target variable
- **Setup Instructions**: Step-by-step guide to set up the environment
- **How to Run**: Commands to train model, run predictions, start service
- **Project Structure**: Overview of files and folders
- **Results/Metrics**: Model performance metrics
- **Deployment**: How to access the deployed service (if applicable)

**Template Structure**:
```markdown
# Project Title

## Problem Description
[Clear description of what problem you're solving]

## Dataset
- Source: [where data comes from]
- Size: [number of records, features]
- Target variable: [what you're predicting]

## Installation

### Prerequisites
- Python 3.x
- Docker (for containerization)

### Setup
```bash
# Clone repository
git clone [repo-url]

# Install dependencies
pipenv install
# or
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py
```

### Running the Service
```bash
python predict.py
```

### Using Docker
```bash
docker build -t ml-service .
docker run -p 9696:9696 ml-service
```

## Project Structure
```
├── README.md
├── notebook.ipynb
├── train.py
├── predict.py
├── model.pkl
├── Dockerfile
├── Pipfile
├── Pipfile.lock
└── data/
```

## Model Performance
[Include key metrics]

## Deployment
[URL or instructions to access deployed service]
```

### 2. Data Management
**Options**:
- **Commit small datasets** (<10MB) directly to repository in `data/` folder
- **Download instructions** for larger datasets with script or clear steps
- **Sample data** for testing if full dataset is too large

**Best Practices**:
```python
# Create a data loading script
# data/download_data.py
import requests
import pandas as pd

def download_data(url, output_path):
    """Download dataset from URL"""
    response = requests.get(url)
    with open(output_path, 'wb') as f:
        f.write(response.content)
    print(f"Data downloaded to {output_path}")

if __name__ == "__main__":
    download_data(
        "https://example.com/dataset.csv",
        "data/dataset.csv"
    )
```

### 3. Notebook (notebook.ipynb)
**Required Sections**:

#### 3.1 Data Preparation & Cleaning
```python
# Load data
import pandas as pd
df = pd.read_csv('data/dataset.csv')

# Check for missing values
df.isnull().sum()

# Handle missing values
df = df.fillna(df.mean())

# Remove duplicates
df = df.drop_duplicates()

# Data type conversions
df['column'] = df['column'].astype('category')
```

#### 3.2 Exploratory Data Analysis (EDA)
```python
# Statistical summary
df.describe()

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Distribution plots
df.hist(figsize=(12, 8))

# Correlation matrix
sns.heatmap(df.corr(), annot=True)

# Target variable analysis
df['target'].value_counts().plot(kind='bar')
```

#### 3.3 Feature Importance Analysis
```python
from sklearn.ensemble import RandomForestClassifier

# Train initial model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Feature importance
importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Visualize
plt.figure(figsize=(10, 6))
sns.barplot(data=importance.head(10), x='importance', y='feature')
plt.title('Top 10 Most Important Features')
```

#### 3.4 Model Selection & Parameter Tuning
```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Try multiple models
models = {
    'logistic': LogisticRegression(),
    'random_forest': RandomForestClassifier(),
    'gradient_boosting': GradientBoostingClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

# Parameter tuning for best model
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(),
    param_grid,
    cv=5,
    scoring='f1_weighted'
)
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
```

### 4. Training Script (train.py)
**Purpose**: Train final model and save it

**Template**:
```python
#!/usr/bin/env python
# train.py

import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    """Load and prepare data"""
    df = pd.read_csv(filepath)
    return df

def prepare_features(df):
    """Feature engineering and preprocessing"""
    # Separate features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Apply transformations
    # ... your preprocessing steps ...
    
    return X, y

def train_model(X_train, y_train):
    """Train the final model"""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def save_model(model, filepath):
    """Save model to disk"""
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {filepath}")

def main():
    # Load data
    print("Loading data...")
    df = load_data('data/dataset.csv')
    
    # Prepare features
    print("Preparing features...")
    X, y = prepare_features(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = train_model(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    save_model(model, 'model.pkl')

if __name__ == "__main__":
    main()
```

**Alternative: Using BentoML**:
```python
import bentoml
from sklearn.ensemble import RandomForestClassifier

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save with BentoML
bentoml.sklearn.save_model(
    "credit_risk_model",
    model,
    signatures={
        "predict": {
            "batchable": True,
        }
    }
)
```

### 5. Prediction Script (predict.py)
**Purpose**: Load model and serve predictions via web service

**Template with Flask**:
```python
#!/usr/bin/env python
# predict.py

import pickle
from flask import Flask, request, jsonify
import pandas as pd

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask('ml-service')

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    # Get data from request
    data = request.get_json()
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # Make prediction
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    
    # Return result
    result = {
        'prediction': int(prediction[0]),
        'probability': float(probability[0][1])
    }
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
```

**Alternative: Using BentoML**:
```python
import bentoml
import pandas as pd
from bentoml.io import JSON, NumpyNdarray

# Load model
model_runner = bentoml.sklearn.get("credit_risk_model:latest").to_runner()

svc = bentoml.Service("credit_risk_classifier", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())
def predict(input_data):
    df = pd.DataFrame([input_data])
    result = model_runner.predict.run(df)
    return {"prediction": int(result[0])}
```

### 6. Dependencies Management

#### Option A: Pipenv (Recommended)
```bash
# Initialize Pipenv
pipenv install

# Install packages
pipenv install pandas scikit-learn flask

# Install dev dependencies
pipenv install --dev jupyter pytest

# This creates Pipfile and Pipfile.lock
```

**Pipfile example**:
```toml
[[source]]
url = "https://pypi.org/simple"
verify_ssl = true
name = "pypi"

[packages]
pandas = "==2.0.0"
scikit-learn = "==1.3.0"
flask = "==2.3.0"
gunicorn = "==21.0.0"

[dev-packages]
jupyter = "*"
pytest = "*"

[requires]
python_version = "3.11"
```

#### Option B: requirements.txt
```txt
pandas==2.0.0
scikit-learn==1.3.0
flask==2.3.0
gunicorn==21.0.0
```

#### Option C: conda environment.yml
```yaml
name: ml-project
channels:
  - defaults
dependencies:
  - python=3.11
  - pandas=2.0.0
  - scikit-learn=1.3.0
  - flask=2.3.0
  - pip:
    - gunicorn==21.0.0
```

### 7. Dockerfile
**Purpose**: Containerize the application

**Template**:
```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy dependency files
COPY Pipfile Pipfile.lock ./

# Install pipenv and dependencies
RUN pip install pipenv && \
    pipenv install --system --deploy

# Copy application files
COPY train.py predict.py ./
COPY model.pkl ./
COPY data/ ./data/

# Expose port
EXPOSE 9696

# Run the service
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
```

**Alternative (with requirements.txt)**:
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

EXPOSE 9696

CMD ["gunicorn", "--bind", "0.0.0.0:9696", "predict:app"]
```

**Build and run**:
```bash
# Build image
docker build -t ml-service .

# Run container
docker run -p 9696:9696 ml-service

# Test
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.0, "feature2": 2.0}'
```

### 8. Deployment

#### Option A: Cloud Deployment (URL)
**Popular platforms**:
- **AWS EC2/ECS/Lambda**
- **Google Cloud Run**
- **Azure Container Instances**
- **Heroku**
- **Railway**
- **Render**

**Example: Deploy to Render**:
1. Push code to GitHub
2. Connect Render to repository
3. Create new Web Service
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn predict:app`
6. Get deployment URL

**Include in README**:
```markdown
## Deployment
Service deployed at: https://ml-service.onrender.com

### Testing the API
```bash
curl -X POST https://ml-service.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.0, "feature2": 2.0}'
```
```

#### Option B: Video/Image Documentation
If not deploying publicly, document:
1. **Local deployment** - Screenshot of service running
2. **API test** - Screenshot/video of making prediction request
3. **Docker deployment** - Terminal output showing container running
4. **Response** - Screenshot of successful API response

**Tools for recording**:
- Loom (video)
- ShareX (screenshots)
- OBS Studio (screen recording)
- Terminal recording: `asciinema`

## Complete Project Checklist

### Documentation
- [ ] README.md with all required sections
- [ ] Problem description is clear
- [ ] Setup instructions are complete
- [ ] Run instructions are tested

### Data
- [ ] Dataset is committed OR download instructions provided
- [ ] Data directory structure is clear
- [ ] Sample data available for testing

### Notebook
- [ ] Data loading and exploration
- [ ] Data cleaning documented
- [ ] EDA with visualizations
- [ ] Feature importance analysis
- [ ] Multiple models compared
- [ ] Parameter tuning performed
- [ ] Final model selection justified

### Training Script
- [ ] Loads data correctly
- [ ] Applies preprocessing
- [ ] Trains model
- [ ] Saves model to file
- [ ] Prints performance metrics
- [ ] Can be run standalone

### Prediction Script
- [ ] Loads saved model
- [ ] Exposes web service
- [ ] Handles JSON input
- [ ] Returns predictions
- [ ] Has health check endpoint
- [ ] Error handling implemented

### Dependencies
- [ ] Pipfile/Pipfile.lock OR
- [ ] requirements.txt OR
- [ ] environment.yml
- [ ] All dependencies listed with versions

### Docker
- [ ] Dockerfile present
- [ ] Builds successfully
- [ ] Runs service correctly
- [ ] Port exposed properly
- [ ] Can make predictions

### Deployment
- [ ] Service deployed to cloud OR
- [ ] Video/screenshots of local deployment
- [ ] Deployment instructions in README
- [ ] API endpoint documented

## Best Practices

### Code Quality
- Use clear variable names
- Add comments for complex logic
- Follow PEP 8 style guide
- Keep functions focused and small

### Version Control
- Commit frequently with clear messages
- Use .gitignore for large files, credentials
- Don't commit model files if very large (use Git LFS or cloud storage)

### Security
- Never commit API keys or credentials
- Use environment variables for secrets
- Validate input data in prediction service

### Performance
- Log prediction requests
- Add request validation
- Consider caching for repeated predictions
- Monitor memory usage

### Testing
```python
# test_predict.py
import pytest
from predict import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    response = client.post('/predict', json={
        'feature1': 1.0,
        'feature2': 2.0
    })
    assert response.status_code == 200
    assert 'prediction' in response.get_json()
```

## Common Pitfalls to Avoid

1. **Missing dependencies** - Always test installation on fresh environment
2. **Hardcoded paths** - Use relative paths or configuration
3. **No error handling** - Wrap predictions in try-except
4. **Large model files** - Use Git LFS or cloud storage
5. **Incomplete README** - Test instructions by following them exactly
6. **No validation** - Validate input data shape and types
7. **Forgotten model file** - Ensure model.pkl is generated by train.py

## Example Project Structure
```
ml-project/
├── README.md
├── notebook.ipynb
├── train.py
├── predict.py
├── model.pkl
├── Dockerfile
├── Pipfile
├── Pipfile.lock
├── .gitignore
├── data/
│   ├── download_data.py
│   └── dataset.csv
├── tests/
│   └── test_predict.py
└── docs/
    ├── deployment_screenshot.png
    └── api_example.png
```

## Quick Start Template

When creating a new ML project using this skill:

1. **Initialize repository**
```bash
mkdir ml-project && cd ml-project
git init
pipenv install pandas scikit-learn flask jupyter
```

2. **Create files**
- README.md (use template above)
- notebook.ipynb
- train.py
- predict.py
- Dockerfile
- .gitignore

3. **Development workflow**
- Explore data in notebook
- Build and test in notebook
- Extract to train.py
- Create predict.py service
- Test locally
- Containerize
- Deploy

4. **Testing before submission**
```bash
# Test training
python train.py

# Test service
python predict.py
# In another terminal:
curl -X POST http://localhost:9696/predict -H "Content-Type: application/json" -d '{"feature1": 1.0}'

# Test Docker
docker build -t ml-service .
docker run -p 9696:9696 ml-service
```

## Additional Resources

- **Deployment**: AWS, GCP, Azure documentation
- **BentoML**: https://docs.bentoml.org
- **Flask**: https://flask.palletsprojects.com
- **Docker**: https://docs.docker.com
- **Pipenv**: https://pipenv.pypa.io
