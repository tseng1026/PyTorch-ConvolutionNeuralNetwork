import os
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow.keras as keras
from   tensorflow.data import Dataset

import Parsing
import Loading
import Model

if __name__=='__main__':
	gpus = tf.GPUOptions(allow_growth = True)
	sess = tf.Session(config = tf.ConfigProto(gpu_options = gpuopt))
	tf.keras.backend.set_session(sess)

	# parsing the arguments
	args = Parsing.Args()
	dataname = args.d
	listname = args.l
	modlname = args.m

	train, valid = Loading.LoadTrn(dataname, listname, "label number")

	imgTrn, labTrn = zip(*train)
	imgVal, labVal = zip(*valid)
	train = Dataset.from_tensor_slices((np.array(imgTrn), np.array(labTrn))).batch(32)
	valid = Dataset.from_tensor_slices((np.array(imgVal), np.array(labVal))).batch(32)
	print ("[Done] Loading all data (training and validation)!")

	# define loss function and optimizer
	model = Model.Net()

	criterion = "categorical_crossentropy"
	optimizer = "adam"
	print ("[Done] Initializing model and all parameters!")

	# train and validate the model
	model.compile(loss = criterion, optimizer = optimizer, metrics = ['accuracy'])
	model.fit(train, epochs = "epoch", steps_per_epoch = 1, batch_size = 32, validation_data = valid)

	model.save_weights(modlname)