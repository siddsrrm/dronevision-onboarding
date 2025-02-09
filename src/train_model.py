import json
import os
import shutil
from pathlib import Path
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from sklearn.model_selection import train_test_split


class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
       
    def __len__(self):
        return len(self.image_paths)
   
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
       
        if self.transform:
            image = self.transform(image)
           
        return image, label


def prepare_data(annotations_file, images_dir):
    """Prepare data from annotations file and images directory."""
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
   
    class_names = ['commercial', 'nature', 'other', 'residential', 'roads']
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
   
    image_paths = []
    labels = []
   
    for ann in annotations:
        if 'choice' in ann:  # Only include annotated images
            img_path = os.path.join(images_dir, os.path.basename(ann['image']))
            if os.path.exists(img_path):
                image_paths.append(img_path)
                labels.append(class_to_idx[ann['choice']])
   
    return image_paths, labels, class_names


def create_model(num_classes):
    """Create a pre-trained ResNet50 model with modified final layer."""
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
   
    model = model.to(device)
    best_val_acc = 0
   
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
       
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
           
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
           
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
       
        train_acc = 100. * correct / total
       
        # Validation phase
        model.eval()
        correct = 0
        total = 0
       
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
       
        val_acc = 100. * correct / total
       
        print(f'Epoch [{epoch+1}/{num_epochs}] - '
              f'Train Loss: {running_loss/len(train_loader):.3f}, '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%')
       
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
       
        scheduler.step()
   
    return model


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
   
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
   
    # Only resize and normalize for validation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
   
    # Prepare data
    image_paths, labels, class_names = prepare_data('sky_classification_export/sky_classification_export.json', 'sky_classification_export/images')
   
    # Split data into train and validation sets
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
   
    # Create datasets
    train_dataset = ImageClassificationDataset(train_paths, train_labels, train_transform)
    val_dataset = ImageClassificationDataset(val_paths, val_labels, val_transform)
   
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
   
    # Create and train model
    model = create_model(num_classes=len(class_names))
    trained_model = train_model(model, train_loader, val_loader, num_epochs=5, device=device)
   
    print("Training completed! Best model saved as 'best_model.pth'")


if __name__ == "__main__":
    main()
