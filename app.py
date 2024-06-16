import sys
import os
import sounddevice as sd
import wavio
import librosa
import numpy as np
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

app = Flask(__name__)


# Load audio files from dataset
def load_audio_files(data_dir):
    audio_data = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".ogg"):
                file_path = os.path.join(root, file)
                label = 1 if "positive" in root else 0
                audio_data.append((file_path, label))
    return audio_data


# Extract audio features of a file
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)
        ex_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return np.mean(ex_features.T, axis=0)
    except Exception as e:
        raise ValueError(f"Error extracting features: {str(e)}")


# Preprocess user sound
def preprocess_user_sound(file_path):
    mfccs = extract_features(file_path)
    return np.array(mfccs).reshape(1, -1)


# Predict with the trained model
def predict_sickness(user_features, model):
    if user_features is None:
        raise ValueError("No features to predict")
    prediction = model.predict(user_features)
    return 'Sick' if prediction[0] == 1 else 'Healthy'


# Load and train the model
def train_model():
    audio_data = load_audio_files("dataset")
    features = []
    labels = []

    for file_path, label in audio_data:
        ex_features = extract_features(file_path)
        features.append(ex_features)
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))

    joblib.dump(model, 'model.pkl')
    return model


# Load the model at startup
model = train_model()


@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if file is None:
            return jsonify({'error': 'No file uploaded'}), 400

        file_path = 'user_cough.ogg'
        file.save(file_path)

        user_features = preprocess_user_sound(file_path)
        user_prediction = predict_sickness(user_features, model)

        return jsonify({'prediction': user_prediction})

    except ValueError as e:
        print(f'ValueError: {str(e)}')
        return jsonify({'error': str(e)}), 500
    except Exception as e:
        print(f'Error: {str(e)}')
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
