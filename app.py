from PIL import Image
import numpy as np
from flask import Flask, request
import io
import utils
import json
import cv2

global model, labels
model = None
labels = None
model, labels = utils._load_model()

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World"

@app.route("/getImageLabel", methods=["POST"])
def _get_image_label():
    data = {"success": False}
    if request.files.get("image"):
        image = request.files["image"].read()
        image = Image.open(io.BytesIO(image))
        image = utils._preprocessing_image(image)
        pred = model.predict(image)
        pred_label = labels.classes_[np.argmax(pred)]
        data["success"] = True
        data["label"] = pred_label
        data["prediction"] = pred
    return json.dumps(data, ensure_ascii=False, cls=utils.NumpyEncoder)

@app.route("/getVideoLabel", methods=["POST"])
def _get_video_label():
    data = {"success": False}
    step = 50
    mean = np.array([122.34, 125.79, 108.16][::1], dtype="float32")

    if request.files.get("video"):
        video = request.files["video"]
        video_format = video.filename.split(".")[-1]
        filename = "video/video." + video_format
        video.save(filename)
        vs = cv2.VideoCapture(filename)
        num_frame = int(vs.get(cv2.CAP_PROP_FRAME_COUNT))
        num_frame = int(num_frame/step)
        (W, H) = (None, None)

        iter = 0
        res = np.asarray([0, 0, 0], dtype=np.float32)
        while True:
            (grabbed, frame) = vs.read()
            if not grabbed:
                break
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224)).astype("float32")
            frame -= mean
            if iter%step == 0:
                preds = model.predict(np.expand_dims(frame, axis=0))[0]
                res = res + np.asarray(preds, dtype=np.float32) / num_frame
            
            key = cv2.waitKey(1) & 0xFF
            if (key == ord("q")):
                break
            iter = iter + 1
        pred_label = labels.classes_[np.argmax(res)]
        data["success"] = True
        data["label"] = pred_label
        data["frame"] = num_frame
        data["prediction"] = res

    return json.dumps(data, ensure_ascii=False, cls=utils.NumpyEncoder)      

if __name__ == "__main__":
    app.run(debug=False)