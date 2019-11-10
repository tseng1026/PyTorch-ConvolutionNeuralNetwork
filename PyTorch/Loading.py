import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from   torch.utils.data import Dataset
from   torchvision import transforms, datasets

def LoadTrn(img_dataname, lab_dataname):
	import glob
	import random
	img = sorted(glob.glob(os.path.join(img_dataname, "*.jpg")))
	lab = pd.read_csv(lab_dataname)
	lab = lab.iloc[:,1].values.tolist()

	data = list(zip(img, lab))
	random.shuffle(data)

	train = data[:0.8 * len(data)]
	valid = data[0.8 * len(data):]
	return train, valid

def LoadTst(img_dataname):
	import glob
	img = sorted(glob.glob(os.path.join(img_dataname, "*.jpg")))

	data = list(img)
	return data

class DataTrn(Dataset):
	def __init__(self, data):
		self.data      = data
		self.transform = transforms.Compose([
						 # transforms.RandomAffine(15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10, fillcolor=0),
						 # transforms.RandomHorizontalFlip(),
						 transforms.ToTensor(),
						 # transforms.Normalize([mean], [std], inplace=False)
						 ])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, ind):
		img = Image.open(self.data[ind][0])	# .convert("RGB")
		img = self.transform(img)
		lab = self.data[ind][1]
		return img, lab

class DataTst(Dataset):
	def __init__(self, data):
		self.data      = data
		self.transform = transforms.Compose([
						 transforms.ToTensor(),
						 # transforms.Normalize([mean], [std], inplace=False)
						 ])

	def __len__(self):
		return len(self.data)

	def __getitem__(self, ind):
		img = Image.open(self.data[ind])	# .convert("RGB")
		img = self.transform(img)
		return img
