import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Load dữ liệu từ file CSV
data = pd.read_csv('hand_gesture_data.csv')

# Chia dữ liệu thành features (X) và nhãn (y)
X = data.drop('label', axis=1)
y = data['label']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo và huấn luyện mô hình KNN
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

# Đánh giá các chỉ số cho mô hình KNN
accuracy_knn = accuracy_score(y_test, y_pred_knn) * 100
precision_knn = precision_score(y_test, y_pred_knn, average='weighted') * 100
recall_knn = recall_score(y_test, y_pred_knn, average='weighted') * 100
f1_knn = f1_score(y_test, y_pred_knn, average='weighted') * 100

# Khởi tạo và huấn luyện mô hình SVM
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

# Đánh giá các chỉ số cho mô hình SVM
accuracy_svm = accuracy_score(y_test, y_pred_svm) * 100
precision_svm = precision_score(y_test, y_pred_svm, average='weighted') * 100
recall_svm = recall_score(y_test, y_pred_svm, average='weighted') * 100
f1_svm = f1_score(y_test, y_pred_svm, average='weighted') * 100

# So sánh các chỉ số trên biểu đồ cột
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
knn_scores = [accuracy_knn, precision_knn, recall_knn, f1_knn]
svm_scores = [accuracy_svm, precision_svm, recall_svm, f1_svm]

x = range(len(metrics))

# Tạo biểu đồ cột
plt.bar(x, knn_scores, width=0.4, label='KNN', color='blue', align='center')
plt.bar([p + 0.4 for p in x], svm_scores, width=0.4, label='SVM', color='green', align='center')

# Thiết lập biểu đồ
plt.xlabel('Metrics')
plt.ylabel('Score (%)')
plt.title('Comparison of KNN and SVM on Different Metrics')
plt.xticks([p + 0.2 for p in x], metrics)
plt.ylim(0, 100)  # Giới hạn trục y từ 0 đến 100%
plt.grid(True)
plt.legend()

# Thêm chú thích cho mỗi cột
for i in range(len(metrics)):
    plt.text(x[i], knn_scores[i] + 1, f'{knn_scores[i]:.2f}%', ha='center', va='bottom', fontsize=10, color='blue')
    plt.text(x[i] + 0.4, svm_scores[i] + 1, f'{svm_scores[i]:.2f}%', ha='center', va='bottom', fontsize=10, color='green')

# Hiển thị biểu đồ
plt.show()
