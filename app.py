import os
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, render_template, flash
from werkzeug.utils import secure_filename

# Get absolute path for model.h5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

# Check if model exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(BASE_DIR, "uploaded_files")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
app.secret_key = "supersecretkey"  # Needed for flash messages

# Allowed audio file extensions
ALLOWED_EXTENSIONS = {"wav", "mp3", "flac"}
SAMPLE_RATE = 16000
DURATION = 5
TARGET_SAMPLES = SAMPLE_RATE * DURATION


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_features(file_path):
    """Extract audio features for prediction"""
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Trim/pad audio
        if len(audio) > TARGET_SAMPLES:
            audio = audio[:TARGET_SAMPLES]
        elif len(audio) < TARGET_SAMPLES:
            audio = np.pad(audio, (0, TARGET_SAMPLES - len(audio)))

        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=128)
        mel_db = librosa.power_to_db(mel_spec)

        # MFCCs
        mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20)

        # Normalize features
        mel_db = (mel_db - np.min(mel_db)) / (np.max(mel_db) - np.min(mel_db))
        mfcc = (mfcc - np.min(mfcc)) / (np.max(mfcc) - np.min(mfcc))

        # Concatenate features
        features = np.concatenate([mel_db, mfcc], axis=0)
        features = np.expand_dims(features, axis=-1)  # Add channel dimension

        return np.array([features])  # Model expects batch input
    except Exception as e:
        raise ValueError(f"Error processing audio: {str(e)}")


@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("audio")

        if not file or not file.filename:
            flash("No file selected. Please upload an audio file.", "error")
            return {"success": False, "error": "No file selected. Please upload an audio file."}

        if not allowed_file(file.filename):
            flash("Invalid file type. Allowed formats: WAV, MP3, FLAC.", "error")
            return {"success": False, "error": "Invalid file type. Allowed formats: WAV, MP3, FLAC."}

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            features = extract_features(filepath)
            prediction = model.predict(features)[0][0]
            result = "Fake" if prediction > 0.5 else "Real"
            return {"success": True, "prediction": result}
        except Exception as e:
            error_message = f"Error processing file: {str(e)}"
            flash(error_message, "error")
            return {"success": False, "error": error_message}

    return render_template("index.html", result=None)


if __name__ == "__main__":
    app.run(debug=True)
