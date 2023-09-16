import streamlit as st
import numpy as np
import librosa
from keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')  # Replace with the path to your trained model

# Define emotion labels
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to extract MFCC features from an audio file
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

# Streamlit UI
st.title('Speech Emotion Recognition')

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    # Process the uploaded file
    st.audio(uploaded_file, format='audio/wav')

    if st.button('Predict Emotion'):
        # Extract MFCC features from the uploaded audio
        mfcc_features = extract_mfcc(uploaded_file)

        # Prepare the data for prediction
        X_pred = np.expand_dims(mfcc_features, axis=0)
        X_pred = np.expand_dims(X_pred, axis=-1)

        # Make predictions
        predictions = model.predict(X_pred)

        # Convert the predictions to emotion labels
        predicted_emotion_index = np.argmax(predictions)
        predicted_emotion = class_labels[predicted_emotion_index]

        # Display the predicted emotion
        st.write(f'Predicted Emotion: {predicted_emotion}')
