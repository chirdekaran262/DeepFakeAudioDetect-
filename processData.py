import os
import numpy as np
import pandas as pd
import librosa
import h5py
from tensorflow.keras.utils import Sequence

class CelebrityVoiceDataGenerator(Sequence):
    def __init__(self, base_dir, batch_size=32, sample_rate=16000, duration=5):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_samples = sample_rate * duration
        
        # Create dataset structure
        self.files_df = self._create_dataset_df()
        self.indexes = np.arange(len(self.files_df))
    
    def _create_dataset_df(self):
        """Create DataFrame with file paths and labels"""
        data = []
        
        # Add real files
        real_dir = os.path.join(self.base_dir, 'Real')
        for file in os.listdir(real_dir):
            if file.endswith('.wav'):
                data.append({
                    'filepath': os.path.join(real_dir, file),
                    'label': 0,  # 0 for real
                    'speaker': file.split('-')[0],
                    'type': 'real'
                })
        
        # Add fake files
        fake_dir = os.path.join(self.base_dir, 'Fake')
        for file in os.listdir(fake_dir):
            if file.endswith('.wav'):
                source, target = file.replace('.wav', '').split('-to-')
                data.append({
                    'filepath': os.path.join(fake_dir, file),
                    'label': 1,  # 1 for fake
                    'source_speaker': source,
                    'target_speaker': target,
                    'type': 'fake'
                })
        
        return pd.DataFrame(data)
    
    def _extract_features(self, audio):
        """Extract audio features useful for deepfake detection"""
        try:
            # Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio, 
                sr=self.sample_rate,
                n_mels=128
            )
            mel_db = librosa.power_to_db(mel_spec)
            
            # MFCCs
            mfcc = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate,
                n_mfcc=20
            )
            
            # Concatenate features
            features = np.concatenate([mel_db, mfcc], axis=0)
            return features
        
        except Exception as e:
            print(f"Feature extraction error: {str(e)}")
            return np.zeros((148, self.target_samples // 512 + 1))  # Fallback
    
    def __len__(self):
        return int(np.ceil(len(self.files_df) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indexes = self.indexes[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_files = self.files_df.iloc[batch_indexes]
        
        # Initialize batch arrays
        X = np.empty((len(batch_files), 148, self.target_samples // 512 + 1))  # 128 mel + 20 mfcc
        y = np.empty(len(batch_files))
        
        for i, (_, row) in enumerate(batch_files.iterrows()):
            try:
                # Load and trim/pad audio
                audio, _ = librosa.load(row['filepath'], sr=self.sample_rate)
                if len(audio) > self.target_samples:
                    audio = audio[:self.target_samples]
                elif len(audio) < self.target_samples:
                    audio = np.pad(audio, (0, self.target_samples - len(audio)))
                
                # Extract features
                features = self._extract_features(audio)
                X[i] = features
                y[i] = row['label']
                
            except Exception as e:
                print(f"Error processing {row['filepath']}: {str(e)}")
                X[i] = np.zeros((148, self.target_samples // 512 + 1))
                y[i] = row['label']
        
        return X[..., np.newaxis], y  # Reshape for CNN
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

def process_dataset(audio_dir, output_file='processed_celebrity_voices.h5'):
    """Process the entire dataset and save to H5 file"""
    generator = CelebrityVoiceDataGenerator(audio_dir)
    
    with h5py.File(output_file, 'w') as hf:
        # Save file information
        file_info = generator.files_df.to_json()
        hf.attrs['file_info'] = file_info
        
        # Create datasets
        X_shape = (len(generator.files_df), 148, generator.target_samples // 512 + 1, 1)
        X_dset = hf.create_dataset('features', shape=X_shape, dtype='float32')
        y_dset = hf.create_dataset('labels', shape=(len(generator.files_df),), dtype='int8')
        
        # Process all files
        for i in range(len(generator)):
            X_batch, y_batch = generator[i]
            start_idx = i * generator.batch_size
            end_idx = min((i + 1) * generator.batch_size, len(generator.files_df))
            X_dset[start_idx:end_idx] = X_batch[:end_idx-start_idx]
            y_dset[start_idx:end_idx] = y_batch[:end_idx-start_idx]
            print(f"Processed batch {i+1}/{len(generator)}")

if __name__ == "__main__":
    # Process the dataset
    audio_dir = r'C:\Users\KARAN\OneDrive\Desktop\MY PROJECT\DeepFake_Audio\ProjectDeepFake\KAGGLE\AUDIO'
    process_dataset(audio_dir)
