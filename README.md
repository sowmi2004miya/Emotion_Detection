# Emotion-Aware Speech Recognition

This project uses **Whisper** for speech recognition and **librosa** to extract audio features (MFCCs and pitch) for emotion recognition. It combines these techniques to transcribe audio from video files and detect the emotion in speech.

## Features

- **Speech Recognition**: Uses OpenAI's **Whisper** to transcribe audio from MP4 files.
- **Emotion Recognition**: Extracts features like **MFCCs** and **pitch** from the audio, and uses a trained **Support Vector Machine (SVM)** model to classify emotions such as **happy**, **sad**, **angry**, or **neutral**.
- **MP4 to WAV Conversion**: Converts audio from MP4 video files to WAV for processing.
- **Emotion-Aware Transcription**: Combines transcription and emotion classification to provide context for the speech.

## Installation

Install the required dependencies by running the following:

```bash
pip install openai-whisper librosa scikit-learn moviepy
##Usage
1. Convert MP4 to WAV
The program first converts MP4 video files into WAV format using MoviePy.

2. Transcribe Speech
It then uses Whisper to transcribe the audio into text.

3. Extract Audio Features
librosa is used to extract Mel-frequency cepstral coefficients (MFCCs) and pitch features from the audio for emotion classification.

4. Emotion Recognition
A simple SVM classifier is used to classify the emotion of the speaker based on the extracted features.

5. Combine Speech Recognition and Emotion Detection
Finally, the system outputs both the transcribed text and the detected emotion of the speech.

##Example Usage
audio_path = "/path/to/your/video.mp4"  # Replace with your video file path
emotion_aware_speech_recognition(audio_path)
This function will:

Convert the MP4 file to WAV format.

Transcribe the audio using Whisper.

Extract audio features and classify the emotion.

Print the transcription and detected emotion.

##Functions Overview
transcribe_audio(audio_path): Transcribes audio from the given path using Whisper.

extract_audio_features(audio_path): Extracts MFCC and pitch features from the audio for emotion detection.

train_emotion_classifier(): Trains a simple SVM classifier to recognize emotions (happy, sad, angry, neutral).

classify_emotion(features, classifier, le): Classifies the emotion based on extracted features using the trained classifier.

convert_mp4_to_wav(mp4_path, wav_path): Converts MP4 audio to WAV format using MoviePy.

emotion_aware_speech_recognition(mp4_path): The main function that combines all steps to transcribe audio and classify emotion.

##Training the Emotion Classifier
The classifier is trained with random sample data, but for better results, you can train it with a labeled dataset of emotions. Once the model is trained, it is used to predict the emotion from the audio features.

##License
This project is licensed under the MIT License. See the LICENSE file for details.
