import pickle
import numpy as np

def check_training_data():
    try:
        print("Loading training data...")
        with open('prepared_data/train_data.pkl', 'rb') as f:
            X_train, y_train = pickle.load(f)
        print("Training data loaded successfully.")
        
        print("Training Data:")
        print(f"Number of training samples: {len(X_train)}")
        print("Labels in training data:", np.unique(y_train, return_counts=True))
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_training_data()
