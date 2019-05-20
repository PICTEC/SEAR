# SEAR - speech expansion and reconstruction

This repository contains code used in SEAR project, which ended in [Waspaa 2019 submission](https://github.com/PICTEC/SEAR/blob/master/docs/bin/WASPAA-final.pdf)
and continuation of previous attempts for [ICASSP 2019 submission](https://www.cmsworkshops.com/ICASSP2019/Papers/Uploads/Proposals/PaperNum/4956/20181030064602_672782_4956.pdf)

### Code structure

The code is divided between old submissions and the current version. Both are stored in order to preserve methods 
that were used to generate the results, however `/old` is not maintained in any way.

Main library for training is in `sear_lib.py`. `model_list.py` contains specifications of the models used for training.
Notebooks server as entry points. They are written in such a way, as to restart the computation whenever something fails.

### Installation and usage

All necessary packages should be in `requirements.txt` file. For tensorflow, please supply it yourself, as I've installed it from source.

Majority of the notebooks use "Python3.6" kernel. This is due to a system upgrade that bumped up Python version during research. If your IPython doesn't have a kernel for 3.6, make one using e.g. [this tutorial][https://stackoverflow.com/questions/28831854/how-do-i-add-python3-kernel-to-jupyter-ipython]

### Sources:

We've used LibriSpeech database as well as publicly available NOISEX93 database.

### Final words

We plan to utilize some of the discoveries associated with the paper in development of an upgraded model. 

Contact information: `pawel.tomasik@pictec.eu`
