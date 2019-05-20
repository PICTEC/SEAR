"""
Visualizer app takes several images produced by the estimator and shows plots of them
"""

import numpy as np
import sys
import time

import matplotlib

def import_embedded_matplotlib():
    from matplotlib.backends.qt_compat import QtCore, QtWidgets, is_pyqt5
    from matplotlib.figure import Figure
    
    if is_pyqt5():
        from matplotlib.backends.backend_qt5agg import (
            FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
    else:
        from matplotlib.backends.backend_qt4agg import (
            FigureCanvas, NavigationToolbar2QT as NavigationToolbar)


    class VisualizerWidget(QtWidgets.QWidget):
    
        __pyQtSignals__ = ()
    
        def __init__(self, parent = None):
            super().__init__(parent)
            self.layout = QtWidgets.QVBoxLayout(self)
            self.canvas_gt = FigureCanvas(Figure(figsize=(1,1)))
            self.layout.addWidget(self.canvas_gt)
            self.canvas_damaged = FigureCanvas(Figure(figsize=(1,1)))
            self.layout.addWidget(self.canvas_damaged)
            self.canvas_predicted = FigureCanvas(Figure(figsize=(1,1)))
            self.layout.addWidget(self.canvas_predicted)
            self.axis_gt = self.canvas_gt.figure.subplots()
            self.axis_damaged = self.canvas_damaged.figure.subplots()
    
    
    class Visualizer(QtWidgets.QMainWindow):
        def __init__(self, visuals):
            super().__init__()
            self._main = QtWidgets.QWidget()
            self.setCentralWidget(self._main)
            self._main_layout = QtWidgets.QVBoxLayout(self._main)
            self._visualizer_widget = VisualizerWidget()
            self._main_layout.addWidget(self._visualizer_widget)
            print(visuals[0].shape)
            self._visualizer_widget.axis_damaged.imshow(visuals[0])
    
        @staticmethod
        def App(visuals=None, gt=None):
            qapp = QtWidgets.QApplication([])
            visuals = Visualizer.compose(visuals, gt)
            reference = Visualizer(visuals).show() # need ot keep item, otherwise GC kills window
            qapp.exec_()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Kappa")
                sys.exit()
    
        @staticmethod
        def compose(visuals, gt):
            return visuals


class Plotter:
    @staticmethod
    def to_file(data, path):
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        scale_min = min([x.min() for x in data])
        scale_max = max([x.max() for x in data])
        fig = plt.figure(figsize=(6,4), dpi=300)
        fig.add_subplot(len(data), 1, 1)
        plt.imshow(np.flip(data[0].T, 0), aspect='equal', vmin=scale_min, vmax=scale_max)
        fig.add_subplot(len(data), 1, 2)
        plt.imshow(np.flip(data[1].T, 0), aspect='equal', vmin=scale_min, vmax=scale_max)
        if len(data) == 3:
            fig.add_subplot(3, 1, 3)
            plt.imshow(np.flip(data[2].T, 0), aspect='equal', vmin=scale_min, vmax=scale_max)
        fig.subplots_adjust(right=0.8)
        axes_for_colorbar = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        plt.colorbar(cax=axes_for_colorbar)
        plt.savefig(path)



def layers_to_visualize(model):
    lyrs = []
    for lyr in model.layers:
        if any([lyr.name.beginswith(x) for x in ("lstm", "conv", "deconv", "dense")]):
            lyrs.append(lyr)
    return lyrs

def visualize(model, image):
    image = image.reshape(1, *image.shape)
    for layer in layers_to_visualize(model):
        model = Model(model.input, layer)
        model.compile('adam', 'mse')
        images = model.predict(image)[0]
        for i in images.shape[2]:
            plt.imshow(images[:,:,i])
    