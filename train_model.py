import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load the collected data
data = pd.read_csv('hand_gesture_data.csv')
X = data.drop('label', axis=1)
y = data['label']

# Check if data is loaded correctly
print(f'Number of samples: {len(X)}')

# Ensure there are at least two classes
if len(y.unique()) < 2:
    raise ValueError("There should be at least two classes to train the model.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the model
joblib.dump(model, 'hand_gesture_model.pkl')
