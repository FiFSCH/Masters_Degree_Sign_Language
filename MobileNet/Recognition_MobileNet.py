import mediapipe as mp
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision import datasets

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-15, 15)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 26)
torch.nn.init.xavier_uniform_(model.classifier[1].weight)
model.load_state_dict(torch.load('./TunedMobileNet.pth', map_location=torch.device('cpu')))
model.eval()

full_dataset = datasets.ImageFolder(root='../data', transform=transform)

class_names = full_dataset.classes
print(class_names)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = mp_hands.process(image=frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark

            min_x = min_y = float('inf')
            max_x = max_y = -float('inf')

            for landmark in landmarks:
                x = landmark.x * frame.shape[1]
                y = landmark.y * frame.shape[0]

                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

            margin = 30
            x1, y1 = max(0, int(min_x - margin)), max(0, int(min_y - margin))
            x2, y2 = min(frame.shape[1], int(max_x + margin)), min(frame.shape[0], int(max_y + margin))

            try:
                hand_region = frame[y1:y2, x1:x2]
            except ValueError:
                continue

            hand_tensor = transforms.ToTensor()(hand_region).unsqueeze(0)
            hand_tensor = transforms.Resize((224, 224))(hand_tensor)

            with torch.no_grad():
                output = model(hand_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)

                class_name = full_dataset.classes[predicted.item()]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(frame, class_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        mp_drawing.draw_landmarks(frame, hand_landmarks)

    cv2.imshow('frame', frame)

    # Press Q to exit 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
