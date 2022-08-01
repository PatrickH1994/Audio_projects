# Audio_projects

Here, I will post different types of audio projects, such as speech emotion recognition and diarization. Most projects are created either by using the Librosa package or then by downloading existing models.


## Projects:

**Keyword detection:**

In this project I use Valerio Velardo's Youtube lectures to build a keyword detection app. The app uses a convolutional neural network that is trained on a speech command dataset (https://www.tensorflow.org/datasets/catalog/speech_commands) to predict the number a person says on an audio file. I then used flask to build a non-user friendly app to which you can input an audio file and it predicts the number.


**Speech emotion recognition**

In this notebook I predict emotions in longer audio files. The programme uses first Pyannote to perform diarization on the audio file and then a pre-trained speech emotion recognition model to predict the emotion in each segment. Finally, I average the emotions in each segment to calculate the overall emotion in the podcast episode, which functions as the test data.
