import os
from flask import Flask, request, jsonify
from utils import loadCheckPoint, predict, createLabelProcessor, createModel


app = Flask(__name__)
ALLOWED_FILE_TYPES = ["avi", "mp4"]


def checkFileType(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_FILE_TYPES


@app.get("/")
def hello_world():
    return "Hello"


@app.route("/predict", methods=["POST"])
def predict_route():
    if (
        "video" not in request.files
        or request.files["video"].filename == ""
        or not checkFileType(request.files["video"].filename)
    ):
        return "Video not found"
    video = request.files["video"]
    labelProcessor = createLabelProcessor()
    model = createModel(labelProcessor)

    try:
        filename = video.filename
        video.save(str(filename))
        result = predict(str(filename), model, labelProcessor)
        os.remove(str(filename))
        return jsonify({"result": result})
    except Exception as e:
        print(e)
        return "Something went wrong!"
