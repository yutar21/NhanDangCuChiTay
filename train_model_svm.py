import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Tải dữ liệu từ file CSV
data = pd.read_csv('hand_gesture_data.csv')

# Tách dữ liệu thành các đặc trưng (X) và nhãn (y)
X = data.drop('label', axis=1)  # Các đặc trưng, loại bỏ cột 'label'
y = data['label']  # Cột nhãn

# Kiểm tra số lượng mẫu trong dữ liệu
print(f'Số lượng mẫu: {len(X)}')  # In số lượng mẫu trong dữ liệu

# Đảm bảo dữ liệu có ít nhất hai lớp khác nhau
if len(y.unique()) < 2:
    raise ValueError("Dữ liệu cần có ít nhất hai lớp để huấn luyện mô hình.")  # Kiểm tra nếu có ít hơn hai lớp

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình SVM với kernel tuyến tính
model = SVC(kernel='linear')

# Huấn luyện mô hình trên tập huấn luyện
model.fit(X_train, y_train)

# Dự đoán nhãn cho tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình bằng độ chính xác
print(f'Độ chính xác: {accuracy_score(y_test, y_pred)}')  # In ra độ chính xác của mô hình

# Lưu mô hình đã huấn luyện vào file
joblib.dump(model, 'hand_gesture_model.pkl')
