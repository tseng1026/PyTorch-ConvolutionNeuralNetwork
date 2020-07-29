# Utils - Convolution Neural Network

## Basic Execution
- **Platform:** Linux (Workstation)
- **Language:** Python3
- **Environment:** GPU
- **Framework:** Tensorflow/Keras
- **Usage:**
	- ``CUDA_VISIBLE_DEVICES=<number> python Train.py -d <data dir> -l <data list> -m <model name> ``
	- ``CUDA_VISIBLE_DEVICES=<number> python Test.py -d <data dir> -m <model name> -o <output file>``
- **Requirements:**
	- python 3.6
	- tensorflow 2.0 ``pip install tensorflow``
	- pillow 6.1		``pip install pillow``
	- matplotlib 3.1	``pip install matplotlib``
