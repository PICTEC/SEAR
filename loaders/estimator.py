import dill
import keras
import os
import tensorflow as tf # for importing reasons
import warnings

from loaders.custom import custom_objects
from loaders.visual import Plotter

def unpickle(path):
    with open(path, "rb") as f:
        return dill.load(f)

def enpickle(what, path):
    with open(path, "wb") as f:
        return dill.dump(what, f)


class DefaultVisualizeTransform:
    def __init__(self):
        pass

    def __call__(self, data):
        return data

class Estimator:
    def __init__(self, visualizer_transform, reverse_transform, dataset_prototype, model):
        self.visualizer_transform = visualizer_transform
        self.reverse_transform = reverse_transform
        self.dataset_prototype = dataset_prototype
        self.model = model

    @staticmethod
    def only_model(path):
        print(os.listdir(path))
        if os.path.exists(os.path.join(path, "weights.h5")):
            with open(os.path.join(path, "arch.json"), "r") as f: 
                jsontext = f.read()
            model = keras.models.model_from_json(jsontext)
            model.load_weights(os.path.join(path, "weights.h5"))
        else:
            if os.path.exists(os.path.join(path, "test_model.h5")):
                model_path = os.path.join(path, "test_model.h5")
            else:
                model_path = os.path.join(path, "model.h5")
            model = keras.models.load_model(model_path, custom_objects)
        return model

    @staticmethod
    def from_folder(path):
        model = Estimator.only_model(path)
        reverse_transform = unpickle(os.path.join(path, "inverse_transform.dill"))
        visualizer_transform = unpickle(os.path.join(path, "visual_transform.dill"))
        dataset_prototype = unpickle(os.path.join(path, "dataset_prototype.dill"))
        return Estimator(visualizer_transform, reverse_transform, dataset_prototype, model)

    def prepare_visualizations(self, examples, groundtruth):
        if self.visualizer_transform is None:
            warnings.warn("Visualizer transform is not defined, falling back to default")
            self.visualizer_transform = DefaultVisualizeTransform()
        if groundtruth is not None:
            dataset = self.dataset_prototype.from_files(groundtruth)
            examples = dataset.produce_single()[0]  # is it dataset class of derivative?
        inputs, outputs = self.process(examples, with_examples=True)
        if groundtruth:
            return [
                (self.visualizer_transform(inputs[x,:,:]),
                 self.visualizer_transform(outputs[x,:,:]),
                 self.visualizer_transform(examples[x,:,:]))
                for x in range(outputs.shape[0])]
        return [
            (self.visualizer_transform(inputs[x,:,:]), self.visualizer_transform(outputs[x,:,:]))
            for x in range(outputs.shape[0])]


    def transform(self, examples):
        if self.reverse_transform is None:
            raise RuntimeError("Reconstruction transform is not defined")
        outputs = self.process(examples)
        print(outputs.shape)
        return [self.reverse_transform(outputs[x,:,:]) for x in range(outputs.shape[0])]

    def process(self, examples, with_examples=False):
        dataset = self.dataset_prototype.from_files(examples)
        examples = dataset.produce_single()[0]  # is it dataset class of derivative?
        if with_examples:
            return examples, self.model.predict(examples)
        return self.model.predict(examples)

    def save(self, path, save_model=False):
        enpickle(self.visualizer_transform, os.path.join(path, "visual_transform.dill"))
        enpickle(self.reverse_transform, os.path.join(path, "inverse_transform.dill"))
        enpickle(self.dataset_prototype, os.path.join(path, "dataset_prototype.dill"))
        if save_model:
            self.model.save(os.path.join(path, "model.h5"))

    def plot_to_file(self, input_paths, output_paths, gt_paths):
        vis = self.prepare_visualizations(input_paths, gt_paths)
        for v, of in zip(vis, output_paths):
            Plotter.to_file(v, of)
