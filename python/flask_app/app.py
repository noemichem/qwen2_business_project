from flask import Flask, request, jsonify, send_file, render_template
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging  # Import logging module
from service import preprocess_data
from qwen2_service import initialize_and_load_model

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG,  # Set the logging level
                    format='%(asctime)s - %(levelname)s - %(message)s',  # Format for log messages
                    handlers=[
                        logging.FileHandler("app.log"),  # Save logs to a file
                        logging.StreamHandler()  # Also output to console
                    ])

# Define the base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Set upload, processed, and images folder paths
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
PROCESSED_FOLDER = os.path.join(BASE_DIR, "processed")
IMAGES_FOLDER = os.path.join(BASE_DIR, "static/images")  # Directory for images

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['IMAGES_FOLDER'] = IMAGES_FOLDER  # Add images folder to config

# Ensure the folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(IMAGES_FOLDER, exist_ok=True)  # Ensure images folder exists

@app.route('/')
def index():
    logging.debug("Rendering index.html")  # Logging message
    return render_template('index.html')

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

@app.route('/upload', methods=['POST'])
def upload_file():
    logging.debug("Upload request received")  # Logging message

    if 'file' not in request.files:
        logging.warning("No file part in request")  # Warning log
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        logging.warning("No file selected")  # Warning log
        return jsonify({'error': 'No selected file'}), 400

    # Save the file
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    logging.debug(f"Saving file to {file_path}")  # Logging message
    file.save(file_path)

    try:
        df = pd.read_csv(file_path)  # Read the uploaded file
        df_preprocessed = preprocess_data(file_path)  # Use the preprocessing function
        logging.info("Data processed successfully")  # Info log

        # Selezionare solo le colonne numeriche
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()

        # Crea la cartella per salvare le immagini se non esiste
        before_chart_path = os.path.join(app.config['IMAGES_FOLDER'], 'before_boxplot.png')
        after_chart_path = os.path.join(app.config['IMAGES_FOLDER'], 'after_boxplot.png')

        # Boxplot per i dati prima della pulizia
        logging.debug(f"Generating boxplot for data before cleaning: {before_chart_path}")  # Log di debug
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df[numeric_columns], palette='Set2')  # Usa seaborn per il boxplot
        plt.title('Boxplot delle Feature Numeriche Prima della Pulizia')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(before_chart_path)
        plt.close()

        # Boxplot per i dati dopo la pulizia
        logging.debug(f"Generating boxplot for data after cleaning: {after_chart_path}")  # Log di debug
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=df_preprocessed[numeric_columns], palette='Set2')  # Usa seaborn per il boxplot
        plt.title('Boxplot delle Feature Numeriche Dopo della Pulizia')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(after_chart_path)
        plt.close()

        # Return the filenames of the generated plots
        return jsonify({
            'beforeChart': 'images/before_boxplot.png',  # Adjusted to reflect static path
            'afterChart': 'images/after_boxplot.png',    # Adjusted to reflect static path
            'filename': os.path.basename(file_path)
        })

    except Exception as e:
        logging.error(f"Error processing file: {e}")  # Error log
        return jsonify({'error': f'File processing failed: {e}'}), 500

@app.route('/processed/<filename>', methods=['GET'])
def download_file(filename):
    # Construct the full file path
    file_path = os.path.join(app.config['PROCESSED_FOLDER'], filename)
    logging.debug(f"Download request for file: {file_path}")  # Debug log

    # Check if the file exists before sending
    if os.path.exists(file_path):
        logging.info(f"File found, sending: {file_path}")  # Info log
        return send_file(file_path, as_attachment=True)
    else:
        logging.warning(f"File not found: {file_path}")  # Warning log
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    logging.info("Initializing and loading the model...")  # Info log
    initialize_and_load_model()
    logging.info("Model loaded successfully")  # Info log
    logging.info("Starting Flask server...")  # Info log
    app.run(debug=True)