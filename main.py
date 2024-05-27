from utils import loadCheckPoint, predict, createLabelProcessor, createModel

labelProcessor = createLabelProcessor()
model = createModel(labelProcessor)
# model = loadCheckPoint(model, "/cp/checkpoint")
model = loadCheckPoint(model, "modelh5/video_classification.h5")
predict("AI538.mp4", model, labelProcessor)
