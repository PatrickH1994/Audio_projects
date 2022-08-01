"""
Structure of program

1. Download wav file
2. Calculate MFCC for each file --> 13 coefficients per time step
3. Save the MFCC and the label to a JSON file

Created by: Patrick Hallila 30/07/2022

"""

import enum
import librosa
import os
import json

DATASET_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//VSCode//KeyWord_system//augmented_dataset//augmented_dataset"
JSON_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//VSCode//KeyWord_system//data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec worth of sound

def prepare_dataset(dataset_path = DATASET_PATH, json_path = JSON_PATH, n_mfcc=13, hop_length=512, n_fft=2048):

    #Create data dictionary
    data = {
        "mappings":[], #values: upp, down, ...
        "labels":[], #values: 1,2, ... --> 1 would mean upp
        "MFCCs":[],
        "files": []
    }

    # Loop through all the sub-directories
    for i, (dirpath, dirname, filenames) in enumerate(os.walk(dataset_path)):
        
        #Check that we are not at the root level
        if dirpath is not dataset_path:
            
            #Update the mappings
            category = dirpath.split("//")[-1] #THis may need to change
            data['mappings'].append(category)

            print(f"Processing {category}, with label: {i-1}")

            #loop through the filings
            for f in filenames:
                
                #get filepath
                file_path = os.path.join(dirpath, f)

                #Load audio file
                signal, sr = librosa.load(file_path)

                # Ensure the audio file is at least 1 sec
                if len(signal) >= SAMPLES_TO_CONSIDER:

                    # Enforce 1 sec long signal --> we only take the 1st sec of the audio file
                    signal = signal[:SAMPLES_TO_CONSIDER]

                    # Extract the MFCCs
                    mfcc = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
                    # Store data
                    data['labels'].append(i-1)
                    data["MFCCs"].append(mfcc.T.tolist()) # Converts numpy array to list, so that it can be stored in a json file
                    data["files"].append(file_path)
                    # print(f"{file_path}: {i-1}")
    
    # store in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


if __name__=="__main__":
    prepare_dataset()