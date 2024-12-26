import torch
import torch.nn as nn
from torchvision import transforms, datasets
from Auxilary.Live_camera_footage_capture import capture_camera_footage


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get class names
class_names = datasets.ImageFolder(root='./data', transform=transform).classes
number_of_classes = len(class_names)

# Load pre-trained MobileNet and tuned weights
model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, number_of_classes)
model.load_state_dict(torch.load('./TunedMobileNet.pth', map_location=torch.device('cpu')))
model.eval()

capture_camera_footage(transform, model, class_names)