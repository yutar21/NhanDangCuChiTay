import cv2
import numpy as np
import joblib
import mediapipe as mp 

# Load mô hình KNN đã huấn luyện
knn_model = joblib.load('hand_gesture_model_knn.pkl')

# Initialize MediaPipe và OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Xử lý khung hình để phát hiện các landmarks của tay
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Trích xuất tọa độ landmarks và chuẩn bị dữ liệu cho KNN
            landmarks = []
            for landmark in hand_landmarks.landmark:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
            landmarks = np.array(landmarks).reshape(1, -1)

            # Dự đoán cử chỉ với mô hình KNN
            try:
                gesture = knn_model.predict(landmarks)
                cv2.putText(frame, str(gesture[0]), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            except Exception as e:
                print(f"Error predicting gesture: {e}")

    # Hiển thị video với dự đoán cử chỉ
    cv2.imshow('Hand Gesture Recognition', frame)
    
    # Nhấn 'E' để thoát chương trình
    if cv2.waitKey(10) & 0xFF == ord('e'):
        break

cap.release()
cv2.destroyAllWindows()
