import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import pickle

def load_data(dataset_path, image_size=(128, 128)):
    images = []
    labels = []
    label_dict = {}
    label_counter = 0

    for root, dirs, files in os.walk(dataset_path):
        for dir_name in dirs:
            label_path = os.path.join(root, dir_name)
            label_dict[dir_name] = label_counter
            print(f"Processing folder: {dir_name}")  # Debugging statement
            for image_file in os.listdir(label_path):
                image_path = os.path.join(label_path, image_file)
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = image.resize(image_size)
                    images.append(np.array(image))
                    labels.append(label_counter)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
            label_counter += 1

    images = np.array(images)
    labels = np.array(labels)

    # Normalize images
    images = images / 255.0

    return images, labels, label_dict

def prepare_data():
    try:
        dataset_path = 'processed'
        image_size = (128, 128)
        images, labels, label_dict = load_data(dataset_path, image_size)

        print(f"Total images: {len(images)}")  # Debugging statement
        print(f"Labels distribution: {np.unique(labels, return_counts=True)}")  # Debugging statement

        # Split data into training and validation sets
        test_size = 0.2
        X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=test_size, random_state=42)

        print(f"Training data: {len(X_train)} samples")  # Debugging statement
        print(f"Validation data: {len(X_val)} samples")  # Debugging statement

        # Save data
        data_folder = 'prepared_data'
        os.makedirs(data_folder, exist_ok=True)

        with open(os.path.join(data_folder, 'train_data.pkl'), 'wb') as f:
            pickle.dump((X_train, y_train), f)
        with open(os.path.join(data_folder, 'val_data.pkl'), 'wb') as f:
            pickle.dump((X_val, y_val), f)
        with open(os.path.join(data_folder, 'label_dict.pkl'), 'wb') as f:
            pickle.dump(label_dict, f)

        print("Data preparation completed.")
    except Exception as e:
        print(f"Error during data preparation: {e}")

if __name__ == "__main__":
    prepare_data()
