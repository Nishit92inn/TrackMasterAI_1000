from flask import Flask, render_template, request, jsonify
import os
import threading
import time
from mtcnn import MTCNN
import cv2
from data_preparation import prepare_data
import subprocess
import locale
import requests
from bs4 import BeautifulSoup
import io
import re
import datetime
import json
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle




app = Flask(__name__)

# Create folders if they do not exist
os.makedirs("raw_dataset", exist_ok=True)
os.makedirs("processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("uploads", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Global variables to keep track of progress
progress = 0
current_folder = ""
training_progress = 0
training_log = ""
last_log_length = 0
data_preparation_progress = 0
data_preparation_log = ""
evaluation_progress = 0
evaluation_log = ""
last_log_length = 0


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/image_scraper')
def image_scraper():
    return render_template('image_scraper.html')

@app.route('/face_detection')
def face_detection():
    celebrity_folders = os.listdir('raw_dataset')
    return render_template('face_detection.html', celebrity_folders=celebrity_folders)

@app.route('/check_processed_images', methods=['POST'])
def check_processed_images():
    celebrity_name = request.form['celebrity_name']
    processed_dir = os.path.join("processed", celebrity_name.replace(' ', '_'))
    if os.path.exists(processed_dir) and any("processed_" in fname for fname in os.listdir(processed_dir)):
        return jsonify({"processed": True})
    return jsonify({"processed": False})

@app.route('/start_scraping', methods=['POST'])
def start_scraping():
    global progress
    progress = 0
    celebrity_name = request.form['celebrity_name']
    num_images = int(request.form['num_images'])

    # Start a new thread for the scraping process
    threading.Thread(target=scrape_images, args=(celebrity_name, num_images)).start()

    return jsonify({"status": "started"})

def scrape_images(celebrity_name, num_images):
    global progress
    query = celebrity_name + " celebrity"
    search_url = "https://www.bing.com/images/search?q=" + query + "&form=QBLH&sp=-1&pq=" + query.replace(' ', '+') + "&sc=8-5&sk=&cvid=E0B6AC0EDCD64838BC30E4B4A8B54EDC"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }
    
    response = requests.get(search_url, headers=headers)
    image_urls = []

    # Extract image URLs from the response
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    for a in soup.find_all("a", {"class": "iusc"}):
        try:
            img_json = eval(a["m"])
            image_urls.append(img_json["murl"])
            if len(image_urls) >= num_images:
                break
        except Exception as e:
            continue

    # Create a directory for the celebrity
    celebrity_dir = os.path.join("raw_dataset", celebrity_name.replace(' ', '_'))
    os.makedirs(celebrity_dir, exist_ok=True)

    # Download the images
    for i, img_url in enumerate(image_urls):
        try:
            img_data = requests.get(img_url).content
            img_name = os.path.join(celebrity_dir, f"{celebrity_name.replace(' ', '_')}_{i + 1}.jpg")
            with open(img_name, 'wb') as handler:
                handler.write(img_data)
            progress = int((i + 1) / num_images * 100)
        except Exception as e:
            continue

    progress = 100

@app.route('/start_face_detection', methods=['POST'])
def start_face_detection():
    global progress
    progress = 0
    celebrity_name = request.form['celebrity_name']
    reprocess = request.form['reprocess'] == 'true'

    if not reprocess:
        processed_dir = os.path.join("processed", celebrity_name.replace(' ', '_'))
        if os.path.exists(processed_dir) and any("processed_" in fname for fname in os.listdir(processed_dir)):
            return jsonify({"status": "processed"})
    
    # Start a new thread for the face detection process
    threading.Thread(target=detect_faces, args=(celebrity_name,)).start()

    return jsonify({"status": "started"})

@app.route('/start_face_detection_all', methods=['POST'])
def start_face_detection_all():
    global progress
    global current_folder
    progress = 0
    current_folder = ""

    # Start a new thread for the face detection process for all folders
    threading.Thread(target=detect_faces_all).start()

    return jsonify({"status": "started"})

def detect_faces(celebrity_name):
    global progress
    raw_dir = os.path.join("raw_dataset", celebrity_name.replace(' ', '_'))
    processed_dir = os.path.join("processed", celebrity_name.replace(' ', '_'))
    os.makedirs(processed_dir, exist_ok=True)

    image_files = os.listdir(raw_dir)
    total_images = len(image_files)

    detector = MTCNN()

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(raw_dir, image_file)
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Image not valid")
            results = detector.detect_faces(image)
            for result in results:
                bounding_box = result['box']
                x, y, width, height = bounding_box
                roi = image[y:y+height, x:x+width]
                processed_image_path = os.path.join(processed_dir, f"processed_{celebrity_name.replace(' ', '_')}_{i + 1}.jpg")
                cv2.imwrite(processed_image_path, roi)
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
            continue

        progress = int((i + 1) / total_images * 100)

    progress = 100

def detect_faces_all():
    global progress
    global current_folder
    celebrity_folders = os.listdir('raw_dataset')
    total_folders = len(celebrity_folders)
    detector = MTCNN()

    for folder_index, celebrity_folder in enumerate(celebrity_folders):
        raw_dir = os.path.join("raw_dataset", celebrity_folder)
        processed_dir = os.path.join("processed", celebrity_folder)
        
        current_folder = celebrity_folder

        if os.path.exists(processed_dir) and any("processed_" in fname for fname in os.listdir(processed_dir)):
            continue

        os.makedirs(processed_dir, exist_ok=True)
        image_files = os.listdir(raw_dir)
        total_images = len(image_files)

        for i, image_file in enumerate(image_files):
            image_path = os.path.join(raw_dir, image_file)
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError("Image not valid")
                results = detector.detect_faces(image)
                for result in results:
                    bounding_box = result['box']
                    x, y, width, height = bounding_box
                    roi = image[y:y+height, x:x+width]
                    processed_image_path = os.path.join(processed_dir, f"processed_{celebrity_folder}_{i + 1}.jpg")
                    cv2.imwrite(processed_image_path, roi)
            except Exception as e:
                print(f"Error processing {image_file} in {celebrity_folder}: {e}")
                continue

        progress = int((folder_index + 1) / total_folders * 100)

    progress = 100

@app.route('/progress_data')
def progress_data():
    global progress
    return jsonify({"progress": progress})

@app.route('/overall_progress_data')
def overall_progress_data():
    global progress
    global current_folder
    return jsonify({"progress": progress, "current_folder": current_folder})

@app.route('/train_model')
def train_model():
    return render_template('train_model.html')

@app.route('/start_training', methods=['POST'])
def start_training():
    global training_progress, training_log
    data = request.json
    model = data['model']
    epochs = int(data['epochs'])
    batch_size = int(data['batch_size'])

    # Reset progress and log
    training_progress = 0
    training_log = ""

    # Start a new thread for the training process
    threading.Thread(target=train_model_function, args=(model, epochs, batch_size)).start()

    return jsonify({"status": "started"})

@app.route('/start_data_preparation', methods=['POST'])
def start_data_preparation():
    global data_preparation_progress, data_preparation_log
    data_preparation_progress = 0
    data_preparation_log = ""

    # Start a new thread for the data preparation process
    threading.Thread(target=data_preparation_thread).start()

    return jsonify({"status": "started"})

@app.route('/data_preparation_progress')
def data_preparation_progress_route():
    global data_preparation_progress, data_preparation_log
    return jsonify({"progress": data_preparation_progress, "log": data_preparation_log})

def data_preparation_thread():
    global data_preparation_progress, data_preparation_log
    try:
        prepare_data()
        data_preparation_log += "Data preparation completed.\n"
        data_preparation_progress = 100
    except Exception as e:
        data_preparation_log += f"Error during data preparation: {e}\n"
        data_preparation_progress = 100

def clean_log_output(output):
    ansi_escape = re.compile(r'\x1B[@-_][0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', output)

def train_model_function(model, epochs, batch_size):
    global training_progress, training_log, last_log_length
    training_log = ""
    last_log_length = 0  # Reset the length of the last log

    # Path to the Python interpreter in the virtual environment
    python_executable = os.path.join(os.environ['VIRTUAL_ENV'], 'Scripts', 'python.exe')

    # Set environment variables
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    # Create a new folder for logs if it doesn't exist
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model_logs')
    os.makedirs(log_dir, exist_ok=True)

    # Generate a unique log filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{model}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    # Open the log file
    with open(log_filepath, 'w', encoding='utf-8') as log_file:

        # Call the train_model.py script
        command = [
            python_executable, "train_model.py",
            "--epochs", str(epochs),
            "--batch_size", str(batch_size)
        ]

        # Use the cwd to set the current working directory for the subprocess
        cwd = os.path.dirname(os.path.abspath(__file__))
        
        # Start the subprocess
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd)
        
        # Wrap stdout and stderr in TextIOWrapper with utf-8 encoding
        stdout_wrapper = io.TextIOWrapper(process.stdout, encoding='utf-8')
        stderr_wrapper = io.TextIOWrapper(process.stderr, encoding='utf-8')
        
        # Monitor the progress and logs
        while True:
            output = stdout_wrapper.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                cleaned_output = clean_log_output(output)
                log_file.write(cleaned_output + "\n")
                log_file.flush()
                if "Epoch" in cleaned_output:
                    training_log += cleaned_output.strip().replace("\n", "<br>") + "<br>"
                    training_progress = min(100, training_progress + (100 // epochs))

        stderr = stderr_wrapper.read()
        if stderr:
            cleaned_error = clean_log_output(stderr)
            log_file.write("Error:\n" + cleaned_error + "\n")
            log_file.flush()
            training_log += "Error:<br>" + cleaned_error.replace("\n", "<br>") + "<br>"
        
        training_log += "<br>Training process finished.<br>"
        training_progress = 100

@app.route('/training_progress')
def training_progress_route():
    global training_progress, training_log, last_log_length
    current_log = training_log[last_log_length:]
    last_log_length = len(training_log)
    return jsonify({"progress": training_progress, "log": current_log})

@app.route('/model_evaluation')
def model_evaluation():
    return render_template('model_evaluation.html')

@app.route('/start_evaluation', methods=['POST'])
def start_evaluation():
    global evaluation_progress, evaluation_log
    data = request.form
    model = data['model']
    evaluation_progress = 0
    evaluation_log = ""

    threading.Thread(target=evaluate_model_function, args=(model,)).start()
    
    return jsonify({"status": "started"})

def evaluate_model_function(model):
    global evaluation_progress, evaluation_log

    model_path = os.path.join('models', f'{model}_face_recognition.h5')
    if not os.path.exists(model_path):
        evaluation_log = "Model file not found. Please train the model first.<br>"
        evaluation_progress = 100
        return

    command = [
        os.path.join(os.environ['VIRTUAL_ENV'], 'Scripts', 'python.exe'),
        "evaluate_model.py"
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=os.environ.copy(), cwd=os.path.dirname(os.path.abspath(__file__)))

    while True:
        output = process.stdout.readline()
        if output == "" and process.poll() is not None:
            break
        if output:
            evaluation_log += output.strip().replace("\n", "<br>") + "<br>"
            evaluation_progress = min(100, evaluation_progress + 10)

    stderr = process.communicate()[1]
    if stderr:
        evaluation_log += "Error:<br>" + stderr.replace("\n", "<br>") + "<br>"

    # Read the evaluation metrics
    try:
        log_dir = 'Model_Evaluation_Logs'
        with open(os.path.join(log_dir, 'evaluation_metrics.json'), 'r', encoding='utf-8') as f:
            metrics = json.load(f)
            evaluation_log += f"<br>Evaluation Metrics:<br>Accuracy: {metrics['accuracy']}<br>Precision: {metrics['precision']}<br>Recall: {metrics['recall']}<br>F1 Score: {metrics['f1']}<br>"
            evaluation_log += f"<br>Classification Report:<br>{metrics['report'].replace('\n', '<br>')}<br>"
    except FileNotFoundError:
        evaluation_log += "<br>Evaluation metrics not found.<br>"

    evaluation_log += "<br>Evaluation process finished.<br>"
    evaluation_progress = 100


@app.route('/evaluation_progress')
def evaluation_progress_route():
    global evaluation_progress, evaluation_log, last_log_length
    current_log = evaluation_log[last_log_length:]
    last_log_length = len(evaluation_log)
    return jsonify({"progress": evaluation_progress, "log": current_log})


@app.route('/evaluation_results')
def evaluation_results():
    log_dir = 'Model_Evaluation_Logs'
    try:
        with open(os.path.join(log_dir, 'evaluation_metrics.json'), 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        print("Metrics retrieved:", metrics)  # Debugging statement
        return render_template('evaluation_results.html', metrics=metrics)
    except FileNotFoundError:
        print("No evaluation results found.")  # Debugging statement
        return render_template('evaluation_results.html', metrics=None)


@app.route('/upload_image')
def upload_image():
    return render_template('upload_image.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join('uploads', filename)
        file.save(filepath)
        prediction = make_prediction(filepath)
        return render_template('upload_image.html', prediction=prediction)
    
def make_prediction(image_path):
    model_path = 'models/mobilenetv2_face_recognition.h5'
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(image_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    with open('prepared_data/label_dict.pkl', 'rb') as f:
        label_dict = pickle.load(f)
    class_labels = {v: k for k, v in label_dict.items()}
    
    # Handle case where prediction confidence is low
    confidence = predictions[0][predicted_class_index]
    if confidence < 0.5:  # threshold can be adjusted
        return "Unknown"
    
    predicted_class = class_labels.get(predicted_class_index, "Unknown")
    return predicted_class


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
