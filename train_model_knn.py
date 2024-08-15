import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# Load dữ liệu từ file CSV
data = pd.read_csv('hand_gesture_data.csv')

# Chia dữ liệu thành features (X) và nhãn (y)
X = data.drop('label', axis=1)
y = data['label']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo mô hình KNN
knn_model = KNeighborsClassifier(n_neighbors=5)  # Số lượng hàng xóm gần nhất là 5, có thể điều chỉnh nếu cần

# Huấn luyện mô hình trên tập huấn luyện
knn_model.fit(X_train, y_train)

# Đánh giá mô hình trên tập kiểm tra
y_pred = knn_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# In kết quả đánh giá
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Lưu mô hình để sử dụng sau này
joblib.dump(knn_model, 'hand_gesture_model_knn.pkl')
