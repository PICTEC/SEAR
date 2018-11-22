#!/usr/bin/env python
import argparse
import dill
import keras
import os
import pickle

from loaders.custom import custom_objects

parser = argparse.ArgumentParser(description="View model details")
parser.add_argument("path", help="path to model folder")
path = parser.parse_args().path
mdl = keras.models.load_model(os.path.join(path, "model.h5"), custom_objects)
mdl.summary()
with open(os.path.join(path, "history.pkl"), "rb") as f:
    hist = pickle.load(f)
print(hist)
with open(os.path.join(path, "results.pkl"), "rb") as f:
    res = pickle.load(f)
print(res)
