import cv2
import mediapipe as mp
import numpy as np
import os
import pandas as pd

# Khởi tạo MediaPipe Hands và Drawing utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Danh sách để lưu trữ dữ liệu đặc trưng và nhãn
data = []
labels = []

# Hàm để xử lý hình ảnh và trích xuất đặc trưng tay
def process_image(img_path, label):
    # Đọc hình ảnh từ đường dẫn
    img = cv2.imread(img_path)
    # Chuyển đổi hình ảnh từ BGR sang RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Xử lý hình ảnh để phát hiện tay
    results = hands.process(img_rgb)
    
    # Nếu phát hiện được tay
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            # Trích xuất các điểm đặc trưng của bàn tay
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y) 
            # Thêm đặc trưng và nhãn vào danh sách
            data.append(landmarks)
            labels.append(label)
    else:
        print(f"No hands detected in image: {img_path}")

# Đường dẫn tới thư mục chứa dữ liệu ảnh
input_dir = 'D:\\HocTap\\Python\\NhanDienCuChiTay\\traindata'

# Lặp qua từng ảnh trong thư mục
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    # Lấy ký tự đầu tiên của tên ảnh làm nhãn
    label = img_name[0]  
    # Xử lý ảnh và trích xuất đặc trưng
    process_image(img_path, label)

# Kiểm tra xem dữ liệu đã được thu thập đúng cách chưa
print(f'Number of samples collected: {len(data)}')

# Lưu dữ liệu vào file CSV nếu đã thu thập được dữ liệu
if data:
    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv('hand_gesture_data.csv', index=False)
else:
    print("No data collected. Please check the input directory and images.")
