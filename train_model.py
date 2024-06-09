import argparse
import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

def load_data():
    with open('prepared_data/train_data.pkl', 'rb') as f:
        X_train, y_train = pickle.load(f)
    with open('prepared_data/val_data.pkl', 'rb') as f:
        X_val, y_val = pickle.load(f)
    with open('prepared_data/label_dict.pkl', 'rb') as f:
        label_dict = pickle.load(f)
    return X_train, y_train, X_val, y_val, label_dict

def preprocess_data(X, img_size):
    X_processed = []
    for img in X:
        img = tf.image.resize(img, img_size)  # Resize the image
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)  # Preprocess the image
        X_processed.append(img)
    return np.array(X_processed)

def train_model(epochs, batch_size):
    print("Loading data...")
    X_train, y_train, X_val, y_val, label_dict = load_data()
    img_size = (128, 128)
    
    print(f"Training data: {len(X_train)} samples, Validation data: {len(X_val)} samples")  # Debugging statement
    
    print("Preprocessing data...")
    X_train = preprocess_data(X_train, img_size)
    X_val = preprocess_data(X_val, img_size)
    
    num_classes = len(label_dict)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes)
    
    print(f"Number of classes: {num_classes}")  # Debugging statement

    print("Defining model...")
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    print("Compiling model...")
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    print("Starting training...")
    datagen = ImageDataGenerator()
    
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    validation_generator = datagen.flow(X_val, y_val, batch_size=batch_size)
    
    model.fit(train_generator, validation_data=validation_generator, epochs=epochs)
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    model_path = 'models/mobilenetv2_face_recognition.h5'
    model.save(model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()
    
    train_model(args.epochs, args.batch_size)
