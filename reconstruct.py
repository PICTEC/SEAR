#!/usr/bin/env python
"""
Reconstruct signal using model and griffin-lim
"""

import argparse
import numpy as np
import os
import scipy.io.wavfile as sio
import sys

import loaders.estimator
import loaders.visual

from loaders.extras import set_device
set_device()

DEFAULTMODELPATH = "./model"

parser = argparse.ArgumentParser()
parser.add_argument('--gt', nargs='+', help="groundtruth files - for visualization comparison")
parser.add_argument('--outputs', nargs='+', help="if present, saves processed soundfiles to corresponding files")
parser.add_argument('--visualize', help="whether to run visualization tool", action='store_true')
parser.add_argument('--normalize', help="when outputting sounds, do I normalize them?", action='store_true')
parser.add_argument('--plots', nargs="+", help="files for plots")
parser.add_argument('--model', nargs=1, default=DEFAULTMODELPATH, help="path to model folder")

parser.add_argument('files', nargs='+', help="List of input files to process")
arguments = parser.parse_args(sys.argv[1:])

# you have to use visualize to use gt, if not visualizing,
# you need to output wavs somewhere otherwise the program is pointless
if not arguments.visualize and arguments.outputs is None and not arguments.plots:
    print("Error: you don't visualize files nor save processed files - the invocation is pointless")
    quit()
if arguments.gt is not None and not arguments.visualize and not arguments.plots:
    print("Error: groundtruth files are used only for visualization or plotting purposes")
    quit()   

# verify file counts
if arguments.gt is not None:
    if len(arguments.gt) != len(arguments.files):
        parser.print_usage()
        print("Error: number of groundtruth files is not equal to the processed files")
        quit()

if arguments.outputs:
    if len(arguments.outputs) != len(arguments.files):
        parser.print_usage()
        print("Error: number of target files is not equal to the processed files")
        quit()

if arguments.plots:
    if len(arguments.plots) != len(arguments.files):
        parser.print_usage()
        print("Error: number of target plots is not equal to the processed files")
        quit()

# determine model for predictions - need to have all the feature extractors and stuff...
model_path = arguments.model[0]
try:
    estimator = loaders.estimator.Estimator.from_folder(model_path)
except IOError:
    print("Error: Cannot load estimator from specified path: " + model_path)
    quit()

# make sure all files exist and are valid
errors = []
if arguments.gt is not None:
    for fname in arguments.gt:
        if not os.path.isfile(fname):
            errors.append(fname)
for fname in arguments.files:
    if not os.path.isfile(fname):
        errors.append(fname)
if errors:
    print("The following paths are not valid files:")
    for fname in errors:
        print(fname)
    quit()

# process signals via estimator if outputs are specified
if arguments.outputs is not None:
    sounds = estimator.transform(arguments.files)
    for inp, out in zip(sounds, arguments.outputs):
        print(inp.shape, out)
        if arguments.normalize:
            inp = inp.astype(np.float32)
            inp /= np.max(np.abs(inp))
            inp *= 2**15 - 1
            inp = inp.astype(np.int16)
        sio.write(out, 16000, inp)

# tell estimator to dump plots
if arguments.plots is not None:
    estimator.plot_to_file(arguments.files, arguments.plots, arguments.gt)

# save files to output files
if arguments.visualize:
    loaders.visual.import_embedded_matplotlib()
    visualizables = estimator.prepare_visualizations(arguments.files)
    loaders.visual.Visualizer.App(visualizables, gt=arguments.gt)
