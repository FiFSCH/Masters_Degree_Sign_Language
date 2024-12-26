import torch
import torch.nn as nn
from torchvision import transforms, datasets
from Auxilary.Live_camera_footage_capture import capture_camera_footage


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-15, 15)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get class names
class_names = datasets.ImageFolder(root='../data', transform=transform).classes
number_of_classes = len(class_names)

# Load pre-trained MobileNet and tuned weights
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, number_of_classes)
model.load_state_dict(torch.load('./TunedMobileNet.pth', map_location=torch.device('cpu')))
model.eval()

capture_camera_footage(transform, model, class_names)