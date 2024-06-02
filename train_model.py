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

def load_data(data_path):
    with open(data_path, 'rb') as f:
        data, labels = pickle.load(f)
    return data, labels

def preprocess_data(data, labels, img_size):
    X = []
    y = []

    for img_path, label in zip(data, labels):
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
        X.append(img)
        y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def train_model(data_path, epochs, batch_size):
    print("Loading data...")
    data, labels = load_data(data_path)
    img_size = (128, 128)
    X, y = preprocess_data(data, labels, img_size)
    
    num_classes = len(set(labels))
    y = tf.keras.utils.to_categorical([list(set(labels)).index(label) for label in y], num_classes)
    
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
    datagen = ImageDataGenerator(validation_split=0.2)
    
    train_generator = datagen.flow(X, y, batch_size=batch_size, subset='training')
    validation_generator = datagen.flow(X, y, batch_size=batch_size, subset='validation')
    
    model.fit(train_generator, validation_data=validation_generator, epochs=epochs)
    
    model.save('models/mobilenetv2_face_recognition.h5')
    print("Model saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--batch_size', type=int, required=True)
    args = parser.parse_args()
    
    train_model('train_data.pkl', args.epochs, args.batch_size)
