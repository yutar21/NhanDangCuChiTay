import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

data = []
labels = []

def process_image(img_path, label):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
            data.append(landmarks)
            labels.append(label)
    else:
        print(f"No hands detected in image: {img_path}")

input_dir = 'D:\\HocTap\\Python\\NhanDienCuChiTay\\traindata'

for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    label = img_name[0]  # Assuming label is the first character of the filename
    process_image(img_path, label)

# Check if data is collected correctly
print(f'Number of samples collected: {len(data)}')

# Save the data to a CSV file if data is collected
if data:
    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv('hand_gesture_data.csv', index=False)
else:
    print("No data collected. Please check the input directory and images.")
