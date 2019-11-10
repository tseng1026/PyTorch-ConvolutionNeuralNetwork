# Utils - Convolution Neural Network

## Basic Execution
- **Platform:** Linux (Workstation)
- **Language:** Python3
- **Environment:** GPU
- **Usage:**
	- ``CUDA_VISIBLE_DEVICES=<number> python Train.py -d <data dir> -l <data list> -m <model name> ``
	- ``CUDA_VISIBLE_DEVICES=<number> python Test.py -d <data dir> -m <model name> -o <output file>``
- **Requirements:**
	- python3.6
	- torch 1.2		``pip install torch``
	- torchvision 0.4``pip install torchvision``
	- pillow 6.1		``pip install pillow``
	- matplotlib 3.1	``pip install matplotlib``