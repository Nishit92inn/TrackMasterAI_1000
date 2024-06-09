#This script is used to check the contents of the .pkl files

import pickle
import numpy as np

def check_prepared_data():
    try:
        print("Loading validation data...")
        with open('prepared_data/val_data.pkl', 'rb') as f:
            X_val, y_val = pickle.load(f)
        print("Validation data loaded successfully.")
        
        print("Loading label dictionary...")
        with open('prepared_data/label_dict.pkl', 'rb') as f:
            label_dict = pickle.load(f)
        print("Label dictionary loaded successfully.")
        
        print("Validation Data:")
        print(f"Number of validation samples: {len(X_val)}")
        print("Labels in validation data:", np.unique(y_val))
        
        print("\nLabel Dictionary:")
        for label, index in label_dict.items():
            print(f"{label}: {index}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_prepared_data()
