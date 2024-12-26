import mediapipe as mp
import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands.Hands()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

model = models.squeezenet1_1(pretrained=True)
model.classifier[1] = nn.Conv2d(512, 26, kernel_size=1)
model.num_classes = 26

# Load tuned weights
model.load_state_dict(torch.load('squeezenet_model.pth', map_location=torch.device('cpu')))  # TODO - fix model loading problem
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

            try:
                hand_tensor = transform(hand_region).unsqueeze(0)
            except Exception as e:
                print(f"Error processing image: {e}")
                continue

            with torch.no_grad():
                output = model(hand_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(output, 1)

                class_name = class_names[predicted.item()]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        mp_drawing.draw_landmarks(frame, hand_landmarks)

    cv2.imshow('frame', frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
