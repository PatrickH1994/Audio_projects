"""
This programme is an interface that can be used to make predictions using the key word spotting service.
The programme takes an audio file and predicts which number is said in the audio file.

"""


import tensorflow as tf
import numpy as np
import librosa

MODEL_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//VSCode//KeyWord_system//model.h5"
SAMPLES_TO_CONSIDER = 22050

class _Keyword_Spotting_Service:
    model = None
    _mappings = [
        "eight",
        "five",
        "four",
        "nine",
        "one",
        "seven",
        "six",
        "three",
        "two",
        "zero"
    ]
    # We need this to create a singleton
    _instance = None

    def predict(self, file_path):
        # Exctract the MFCCs
        mfccs = self.preprocess(file_path)

        # Convert 2D mfcc array into 4D array --> (# samples, # segments, # mfcc, # channels)
        mfccs = mfccs[np.newaxis, ..., np.newaxis]

        # Make prediction
        predictions = self.model.predict(mfccs)

        predicted_keyword = self._mappings[np.argmax(predictions)]
        return predicted_keyword
    
    
    def preprocess(self, file_path, n_mfcc=13, hop_length=512, n_fft=2048):
        
        # load audio file
        signal, sr = librosa.load(file_path)

        # ensure consistency in the audio file length
        print(len(signal))
        if len(signal) > SAMPLES_TO_CONSIDER:
            signal = signal[:SAMPLES_TO_CONSIDER]

        # extract mfcc
        mfcc = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

        return mfcc.T


def Keyword_Spotting_Service():
    #Ensure we have only one instance of KSS
    if _Keyword_Spotting_Service._instance is None:
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = tf.keras.models.load_model(MODEL_PATH)
        
    
    return _Keyword_Spotting_Service._instance

if __name__=="__main__":
    kss = Keyword_Spotting_Service()
    k1=kss.predict(file_path="C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//VSCode//KeyWord_system//Test//FOUR.wav") #put test 1
    k2=kss.predict(file_path="C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//VSCode//KeyWord_system//Test//SEVEN.wav") #put test 2

    print(f"Predicted key words: {k1}, {k2}\nCorrect keywords: Four, Seven")