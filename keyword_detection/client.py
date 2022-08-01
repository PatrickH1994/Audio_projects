"""
Example of a client that is interacting with the server.

"""

from email.mime import audio
from urllib import response
import requests


URL = "http://127.0.0.1:5000/predict"
TEST_AUDIO_FILE_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//VSCode//KeyWord_system//Test//FOUR.wav"

if __name__ == "__main__":

    audio_file = open(TEST_AUDIO_FILE_PATH, "rb")
    values = {
        "file": (
            TEST_AUDIO_FILE_PATH,
            audio_file,
            "audio/wav"
        )      
    }

    #Sends a post request
    response = requests.post(URL, files=values)
    data = response.json()
    print(f"Predicted keyword is: {data['keyword']}")