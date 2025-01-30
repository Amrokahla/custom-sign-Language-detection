import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# Load processed data and preprocessing objects
X_train = joblib.load('gesture_data/X_train.pkl')
X_test = joblib.load('gesture_data/X_test.pkl')
y_train = joblib.load('gesture_data/y_train.pkl')
y_test = joblib.load('gesture_data/y_test.pkl')
le = joblib.load('gesture_data/label_encoder.pkl')

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, num_classes=len(le.classes_))
y_test = to_categorical(y_test, num_classes=len(le.classes_))

# Define and train the model
model = Sequential([
    Dense(128, input_shape=(63,), activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(le.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save('gesture_data/model.h5')