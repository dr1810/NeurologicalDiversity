import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import pyaudio
from IPython.display import Audio
from scipy.io import wavfile
import pandas as pd

# Set up PyAudio
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

p = pyaudio.PyAudio()

# Load YAMNet model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load the YAMNet classes
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = list(pd.read_csv(class_map_path)['display_name'])

# Define your custom model classes
my_classes = [
    'dog', 'chirping_birds', 'vacuum_cleaner', 'thunderstorm', 'door_wood_knock',
    'can_opening', 'crow', 'clapping', 'fireworks', 'chainsaw', 'airplane',
    'mouse_click', 'pouring_water', 'train', 'sheep', 'water_drops', 'church_bells',
    'clock_alarm', 'keyboard_typing', 'wind', 'footsteps', 'frog', 'cow', 'brushing_teeth',
    'car_horn', 'crackling_fire', 'helicopter', 'drinking_sipping', 'rain', 'insects',
    'laughing', 'hen', 'engine', 'breathing', 'crying_baby', 'hand_saw', 'coughing',
    'glass_breaking', 'snoring', 'toilet_flush', 'pig', 'washing_machine', 'clock_tick',
    'sneezing', 'rooster', 'sea_waves', 'siren', 'cat', 'door_wood_creaks', 'crickets'
]

# Define your custom model
def create_custom_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape, dtype=tf.float32, name='input_embedding'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ], name='custom_model')

    return model

# Function to ensure sample rate
def ensure_sample_rate(original_sample_rate, waveform, desired_sample_rate=16000):
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = np.interp(
            np.linspace(0, len(waveform) - 1, desired_length),
            np.arange(len(waveform)),
            waveform,
        )
    return desired_sample_rate, waveform

# Function to get MFCC features from audio data
def get_features(audio_data):
    mfccs = librosa.feature.mfcc(y=audio_data, sr=RATE, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

# Function to classify environmental sound using YAMNet
def classify_environmental_sound_yamnet(audio_data):
    audio_data = audio_data.astype(np.float32) / 32767.0
    scores, embeddings, spectrogram = yamnet_model(audio_data)
    class_scores = tf.reduce_mean(scores, axis=0)
    top_class = tf.math.argmax(class_scores)
    inferred_class = class_names[top_class]
    print(f'The main sound is: {inferred_class}')

# Main function to capture live audio and perform classification
def main():
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening... (Press 'q' to quit)")

    try:
        while True:
            # Read audio data from the stream
            audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)

            # Classify environmental sound using YAMNet
            classify_environmental_sound_yamnet(audio_data)

            # Preprocess audio data for custom model
            audio_data = audio_data.astype(np.float32) / 32767.0
            prediction_feature = get_features(audio_data)
            prediction_feature = np.expand_dims(np.array([prediction_feature]), axis=2)

            # Predict using custom model
            custom_model = create_custom_model(input_shape=(40,), num_classes=len(my_classes))
            result = custom_model.predict([prediction_feature])
            predicted_class_index = np.argmax(result)
            predicted_class_label = my_classes[predicted_class_index]
            print(f'Predicted Environmental Sound: {predicted_class_label}')

            # Check for user input to stop listening
            if input("Press 'q' to stop listening: ").lower() == 'q':
                break

    except Exception as e:
        print(f"Error capturing audio: {str(e)}")

    finally:
        # Close the stream when done
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()
