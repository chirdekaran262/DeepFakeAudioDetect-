
# DeepFake Audio Detection Using CNN

DeepFake Audio Detection is a deep learning project designed to identify fake or manipulated audio recordings. The project utilizes a Convolutional Neural Network (CNN) model to distinguish between genuine and fake audio clips.

## Features
- Real-time audio file upload and analysis
- Model training and evaluation using audio datasets
- Audio feature extraction and processing
- Web-based interface for uploading and predicting audio files

## Technologies Used
- **Python**: Core programming language
- **Flask**: Web framework for building the API and UI
- **TensorFlow/Keras**: Deep learning model
- **LibROSA**: Audio processing and feature extraction
- **HTML & CSS**: Frontend design
- **Git**: Version control

## Directory Structure
```
DeepFake_Audio/
├── .idea/                     # IDE configuration files
├── static/                    # Static files (e.g., CSS, JS)
├── templates/                 # HTML templates
├── uploaded_files/            # User-uploaded audio files (auto-created when uploading)
├── app.py                     # Main Flask application
├── model.h5                   # Trained deep learning model (ignored)
├── processData.py             # Data processing and feature extraction
├── processed_celebrit y_voices.h5  # Preprocessed model (ignored)
├── trainModel.py              # Model training script
└── .gitignore                 # Git ignore file
```

```The dataset used in this project is not uploaded to the repository. Please download the dataset from Kaggle and place it in the appropriate folder. Additionally, to run the model, download the pre-trained .h5 file and place it in the project directory.```


## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/username/DeepFake_Audio.git
   cd DeepFake_Audio
   ```
2. Create a virtual environment:
   ```bash
   python -m venv sequence_env
   source sequence_env/bin/activate  # On Windows: sequence_env\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the Flask application:
   ```bash
   python app.py
   ```
5. Access the web application at `http://localhost:5000`

## Usage
1. Upload an audio file via the web interface.
2. Click the 'Predict' button to analyze the audio.
3. View the prediction result to determine whether the audio is fake or real.

## License
This project is licensed under the MIT License. Feel free to use and modify the code as needed.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## Acknowledgements
- Thanks to Kaggle for the dataset.
- Special thanks to the open-source community for their amazing libraries and tools.
