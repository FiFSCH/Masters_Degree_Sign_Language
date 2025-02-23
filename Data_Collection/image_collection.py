import cv2
import os
import time

COLLECTED_DATA_PATH = './../PSL_Collected_Data'
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'W', 'Y','Z']
number_of_img_per_label = 5


def capture_images(label):
    print(f'Currently collecting label: {label}')
    os.makedirs(f'{COLLECTED_DATA_PATH}/{label}', exist_ok=True)
    cap = cv2.VideoCapture(0)
    time.sleep(3)

    for image_number in range(number_of_img_per_label):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image")
            break

        text = f'Label: {label} | Image: {image_number}/{number_of_img_per_label - 1}'
        cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "Press Q to exit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        file_name = f'{COLLECTED_DATA_PATH}/{label}/{label}_{image_number}.jpg'
        cv2.imwrite(file_name, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            exit()

    cap.release()
    cv2.destroyAllWindows()


for letter in labels:
    capture_images(letter)
