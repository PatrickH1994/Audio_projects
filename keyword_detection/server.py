
"""
Work flow:
client --> POST request --> server --> prediction back to client

If I used uwsgi, which I didn't because I couldn't install it, I would use the code below

uwsgi --http 127.0.0.1:5050 --wsgi-file server.py --callable app --processes 1 --threads 1
"""
import random
import os
from flask import Flask, request, jsonify
from keyword_spotting_service import Keyword_Spotting_Service

#Create flask app
app = Flask(__name__)

@app.route("/predict", methods=['POST'])
def predict():
    
    # Get audio file and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0,100000))
    audio_file.save(file_name)

    # Invoke the kss
    kss = Keyword_Spotting_Service()

    # Make a prediction
    predicted_keyword = kss.predict(file_name)

    # Remove the audio file
    os.remove(file_name)

    # Send back the predicted keyword in JSON format
    data = {"keyword":predicted_keyword}
    return jsonify(data)

if __name__=="__main__":
    app.run(debug=False)