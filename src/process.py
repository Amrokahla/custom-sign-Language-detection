from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import joblib

data_path = 'gesture_data/gesture_data.csv'
column_names = ['label'] + [f'lm_{i}' for i in range(63)]  # 21 landmarks * 3 (x, y, z)
data = pd.read_csv(data_path, header=None, names=column_names)

X = data.drop('label', axis=1).values
y = data['label'].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save processed data and preprocessing objects
joblib.dump(le, 'gesture_data/label_encoder.pkl')
joblib.dump(scaler, 'gesture_data/scaler.pkl')
joblib.dump(X_train, 'gesture_data/X_train.pkl')
joblib.dump(X_test, 'gesture_data/X_test.pkl')
joblib.dump(y_train, 'gesture_data/y_train.pkl')
joblib.dump(y_test, 'gesture_data/y_test.pkl')