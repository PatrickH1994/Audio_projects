
"""

Trains a CNN model that predicts which number the person said in the audio file.

"""

import json
import numpy as np
from sklearn.model_selection import train_test_split

#Deep learning libariries
import tensorflow as tf
from tensorflow import keras


DATA_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//VSCode//KeyWord_system//data.json" 
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32 #Number of samples the network will see before it updates the weightings
SAVED_MODEL_PATH = "C://Users//Patrick//OneDrive - City, University of London//PhD//Programming projects//VSCode//KeyWord_system//model.h5"
NUM_OF_KEYWORDS = 10

def load_dataset(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data['MFCCs'])
    y = np.array(data['labels'])

    return X, y

def get_data_splits(data_path, test_size=0.1, test_validation=0.1):
    
    # load dataset
    X, y = load_dataset(data_path)

    # create train/val/test splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=test_validation)

    # convert 2D to 3D arrays
    # Shape prior conversion: (# segments, MFCCs) --> we need (# segments, MFCCs, depth)
    X_train = X_train[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def build_model(input_shape, learning_rate, error="sparse_categorical_crossentropy"):

    # Build the network
    model = keras.Sequential()
    # Conv 1
    model.add(keras.layers.Conv2D(64, (3,3), activation='relu', #(3,3) = kernel size
                input_shape=input_shape, kernel_regularizer=keras.regularizers.L2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2,), padding="same"))

    # Conv 2
    model.add(keras.layers.Conv2D(32, (3,3), activation='relu', #(3,3) = kernel size
                kernel_regularizer=keras.regularizers.L2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((3,3), strides=(2,2,), padding="same"))

    # Conv 3
    model.add(keras.layers.Conv2D(32, (2,2), activation='relu', #(3,3) = kernel size
                kernel_regularizer=keras.regularizers.L2(0.001)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPool2D((2,2), strides=(2,2,), padding="same"))

    # flatten the output
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    #Softmax layer
    model.add(keras.layers.Dense(NUM_OF_KEYWORDS, activation="softmax"))

    # Compile the model
    optimiser = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer = optimiser, loss=error, metrics=["accuracy"])

    # Print model summary
    model.summary()
    
    return model


def main():
    
    # Load train/validation/test splits
    X_train, X_validation, X_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)

    # Build model architecture
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]) # (# segments, # MFCC coefficients, depth) --> we expect (22050, 13, 1) 
    model = build_model(input_shape, LEARNING_RATE)

    # Train model
    model.fit(
        X_train,
        y_train,
        epochs = EPOCHS,
        batch_size = BATCH_SIZE,
        validation_data = (X_validation, y_validation)
    )

    # Evaluate the model
    test_error, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test error: {test_error}\nTest accuracy: {test_accuracy}")

    # Save the model
    model.save(SAVED_MODEL_PATH)

if __name__=="__main__":
    main()