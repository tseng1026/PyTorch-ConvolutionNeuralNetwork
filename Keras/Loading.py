import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from   tensorflow.keras import utils
from   tensorflow.data  import Dataset

def LoadTrn(img_dataname, lab_dataname, num_classes):
	import glob
	import random
	filename = sorted(glob.glob(os.path.join(img_dataname, "*.jpg")))
	img = [Image.open(img) for img in filename]
	lab = pd.read_csv(lab_dataname)
	lab = lab.iloc[:,1].values.tolist()

	img = np.array(imgTrn)
	lab = np.array(labTrn)

	img = img.astype("float32").reshape(-1, 28, 28, 1) / 255
	lab = utils.to_categorical(lab, num_classes)
	
	data = list(zip(img, lab))
	random.shuffle(data)

	train = data[:0.8 * len(data)]
	valid = data[0.8 * len(data):]
	return train, valid

def LoadTst(img_dataname):
	import glob
	filename = sorted(glob.glob(os.path.join(img_dataname, "*.jpg")))
	img = [Image.open(img) for img in filename]

	img = np.array(img)
	img = img.astype("float32").reshape(-1, 28, 28, 1) / 255

	data = list(img)
	return data