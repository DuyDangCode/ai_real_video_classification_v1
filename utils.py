import cv2
from cv2.gapi import mask
from cv2.typing import FeatureExtractor
import numpy as np
import tensorflow as tf
import keras
from tensorflow.python.ops.variables import local_variables


IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 1

MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048


def createLabelProcessor():
    return keras.layers.StringLookup(
        num_oov_indices=0, vocabulary=np.array(["Ai", "Real"])
    )


def cropCenterSquare(frame):
    y, x = frame.shape[0:2]
    minDim = min(y, x)
    startX = (x // 2) - (minDim // 2)
    startY = (y // 2) - (minDim // 2)
    return frame[startY : startY + minDim, startX : startX + minDim]


def loadVideo(videoPath, maxFrames=200, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        return None

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cropCenterSquare(frame)
        frame = cv2.resize(frame, resize)
        frame = frame[:, :, [2, 1, 0]]
        frames.append(frame)
    cap.release()
    return np.array(frames)


def buildFeatureExtractor():
    featureExtrator = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocessInput = keras.applications.inception_v3.preprocess_input
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocessInput(inputs)
    outputs = featureExtrator(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")


def createModel(labelProcessor):
    classVocab = labelProcessor.get_vocabulary()
    frameFeatureInput = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    maskInput = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
    x = keras.layers.GRU(16, return_sequences=True)(frameFeatureInput, mask=maskInput)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(classVocab), activation="softmax")(x)

    rnnModel = keras.Model([frameFeatureInput, maskInput], output)
    rnnModel.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return rnnModel


def loadCheckPoint(model, checkPointPath):
    model.load_weights(checkPointPath)
    return model


def extractFeature(frames, featureExtrator):
    frames = frames[None, ...]
    frameMask = np.zeros(shape=(1, MAX_SEQ_LENGTH), dtype="bool")
    frameFeatures = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        videoLength = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, videoLength)
        for j in range(length):
            frameFeatures[i, j, :] = featureExtrator.predict(batch[None, j, :])
        frameMask[i, :length] = 1
    return frameFeatures, frameMask


def predict(path, model, labelProcessor):
    classVocab = labelProcessor.get_vocabulary()
    frames = loadVideo(path)
    featureExtractor = buildFeatureExtractor()
    framesFeatures, frameMask = extractFeature(frames, featureExtractor)
    result = model.predict([framesFeatures, frameMask])[0]
    for i in np.argsort(result)[::-1]:
        print(f"{classVocab[i]}: {result[i] * 100:5.2f}%")
    if result[0] > result[1]:
        return classVocab[0]
    return classVocab[1]
