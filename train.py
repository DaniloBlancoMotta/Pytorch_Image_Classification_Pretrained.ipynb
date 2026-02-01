#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training Script for PyTorch Image Classification
This script trains a pretrained model for bean leaf disease classification
"""

import torch
from torch import nn
from torch.optim import Adam
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
from PIL import Image
import pickle
import os
from tqdm import tqdm

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class CustomImageDataset(Dataset):
    """Custom Dataset for loading images"""
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_encoder = LabelEncoder()
        self.labels = torch.tensor(
            self.label_encoder.fit_transform(dataframe['category'])
        ).to(device)
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image if there's an error
            return torch.zeros((3, 128, 128)), label

def load_data(data_path='data'):
    """Load and prepare data"""
    print("Loading data...")
    
    # Check if data exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory '{data_path}' not found. Please download the dataset first.")
    
    train_csv = os.path.join(data_path, 'train.csv')
    val_csv = os.path.join(data_path, 'val.csv')
    
    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError("train.csv or val.csv not found in data directory")
    
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    data_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Update paths to be relative to data directory
    data_df['image:FILE'] = data_df['image:FILE'].apply(
        lambda x: os.path.join(data_path, x) if not os.path.isabs(x) else x
    )
    
    print(f"Total samples: {len(data_df)}")
    print(f"Classes distribution:\n{data_df['category'].value_counts()}")
    
    return data_df

def create_model(num_classes=3, model_name='resnet18', pretrained=True):
    """Create and configure pretrained model"""
    print(f"Creating {model_name} model...")
    
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model.to(device)

def train_epoch(model, train_loader, criterion, optimizer):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': running_loss/len(train_loader), 
                         'acc': 100*correct/total})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def evaluate(model, test_loader):
    """Evaluate model on test set"""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    
    return accuracy, all_predictions, all_labels

def save_model(model, label_encoder, filepath='model.pkl'):
    """Save model and preprocessing objects"""
    print(f"\nSaving model to {filepath}...")
    
    model_data = {
        'model_state_dict': model.state_dict(),
        'label_encoder': label_encoder,
        'device': device
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"Model saved successfully!")

def main():
    # Configuration
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001
    MODEL_NAME = 'resnet18'
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load data
    data_df = load_data('data')
    
    # Split data
    train_df, test_df = train_test_split(
        data_df, test_size=0.3, random_state=42, stratify=data_df['category']
    )
    
    print(f"\nTrain samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Create datasets
    train_dataset = CustomImageDataset(train_df, transform=transform)
    test_dataset = CustomImageDataset(test_df, transform=transform)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    
    # Create model
    num_classes = len(data_df['category'].unique())
    model = create_model(num_classes=num_classes, model_name=MODEL_NAME)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch+1}/{EPOCHS}]")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        test_acc, _, _ = evaluate(model, test_loader)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, train_dataset.label_encoder, 'best_model.pkl')
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    # Final evaluation
    print("\nFinal Evaluation on Test Set:")
    test_acc, predictions, labels = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    class_names = train_dataset.label_encoder.classes_
    print(classification_report(labels, predictions, target_names=[str(c) for c in class_names]))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(labels, predictions)
    print(cm)
    
    # Save final model
    save_model(model, train_dataset.label_encoder, 'model.pkl')
    
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
