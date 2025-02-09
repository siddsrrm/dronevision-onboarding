import torch
from torchvision import models, transforms
from PIL import Image



class ResNetClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet50(weights=None)  # Use weights=None instead of pretrained=False
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, num_classes)


    def forward(self, x):
        return self.resnet(x)



num_classes = 5 
model = ResNetClassifier(num_classes=num_classes)



checkpoint = torch.load('best_model.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), weights_only=True)


# Inspect the checkpoint keys
print("Checkpoint keys:", checkpoint.keys())


# Load the state dictionary based on the checkpoint contents
if 'state_dict' in checkpoint:
    model.load_state_dict(checkpoint['state_dict'])
elif 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)  # Directly load the weights


model.eval()  # Set the model to evaluation mode


# Define the preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to the input size expected by the model
    transforms.ToTensor(),          # Convert to tensor
    transforms.Normalize(           # Normalize with the same values used during training
        mean=[0.485, 0.456, 0.406],  # Mean for ImageNet
        std=[0.229, 0.224, 0.225]    # Std for ImageNet
    )
])


# Load and preprocess the image
image_path = '/content/dronevision-onboarding/src/sky_classification_export/images/00247c59-Data0_000280_800_3200.jpg'  # Replace with the path to your image
image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
input_tensor = preprocess(image).unsqueeze(0)  # Add a batch dimension


# Move the input tensor to the same device as the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_tensor = input_tensor.to(device)
model = model.to(device)


# Run inference
with torch.no_grad():  # Disable gradient calculation
    output = model(input_tensor)


# Get the predicted class
_, predicted_class = torch.max(output, 1)
predicted_class = predicted_class.item()  # Convert to a Python integer


# Define class labels (must match the order used during training)
class_labels = ['residential', 'commercial', 'nature', 'roads', 'other']


# Get the predicted class label
predicted_label = class_labels[predicted_class]
print(f"Predicted Class: {predicted_label}")
