import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import pickle
import numpy as np
import os
import json

def load_data():
    with open('prepared_data/val_data.pkl', 'rb') as f:
        X_val, y_val = pickle.load(f)
    with open('prepared_data/label_dict.pkl', 'rb') as f:
        label_dict = pickle.load(f)
    return X_val, y_val, label_dict

def evaluate_model():
    model_path = 'models/mobilenetv2_face_recognition.h5'
    if not os.path.exists(model_path):
        print("Model file not found. Please train the model first.")
        return

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("Loading data...")
    X_val, y_val, label_dict = load_data()

    print(f"Validation data: {len(X_val)} samples")
    print(f"Labels in validation data: {np.unique(y_val, return_counts=True)}")
    print(f"Label dictionary: {label_dict}")

    print("Preprocessing data...")
    X_val = np.array([tf.image.resize(img, (128, 128)) for img in X_val])
    X_val = tf.keras.applications.mobilenet_v2.preprocess_input(X_val)

    print("Evaluating model...")
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_val, y_pred_classes)
    precision = precision_score(y_val, y_pred_classes, average='weighted')
    recall = recall_score(y_val, y_pred_classes, average='weighted')
    f1 = f1_score(y_val, y_pred_classes, average='weighted')

    report = classification_report(y_val, y_pred_classes, target_names=label_dict.keys())

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report
    }

    log_dir = 'Model_Evaluation_Logs'
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, 'evaluation_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f)

    print("Evaluation metrics saved.")

if __name__ == "__main__":
    evaluate_model()
