import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from Auxilary.Live_camera_footage_capture import capture_camera_footage

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Get class names
class_names = datasets.ImageFolder(root='./data', transform=transform).classes
number_of_classes = len(class_names)

# Load pre-trained SqueezeNet and tuned weights
model = models.squeezenet1_0(weights=models.SqueezeNet1_0_Weights.DEFAULT)
model.classifier[1] = nn.Conv2d(512, number_of_classes, kernel_size=1)
model.load_state_dict(torch.load('squeezenet_model_old.pth', map_location=torch.device('cpu')))
model.eval()

capture_camera_footage(transform, model, class_names)


