from numpy.lib.arraysetops import isin
from tensorflow.keras.models import load_model
import numpy as np
import pickle
from PIL import Image
import json

def _load_model():
    model = load_model("model/activity.model")
    labels = pickle.loads(open("model/lb.pickle", "rb").read())
    return model, labels

def _preprocessing_image(image):
    mean = np.array([122.34, 125.79, 108.16][::1], dtype="float32")
    img = image.resize((224, 224))
    img = np.asarray(img)
    img = img - mean
    img = np.expand_dims(img, axis=0)
    return img

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32):
            return int(obj)
        if isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.float64):
            return float(obj)
        return json.JSONEncoder.default(self, obj)
