import os
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from   tensorflow.data import Dataset

import Parsing
import Loading
import Model

if __name__=='__main__':
	# parsing the arguments
	args = Parsing.Args()
	dataname = args.d
	modlname = args.m
	outputfile = args.o

	test = Loading.LoadTst(dataname)
	numb = len(test)

	test = Dataset.from_tensor_slices(np.array(test)).batch(32)
	print ("[Done] Loading all data (testing)!")
	
	model = Model.Net()
	model.load_weights(modlname)
	predt = model.predict(test)

	# write the results to file
	index = np.arange(numb)
	index = index.astype("int")

	predict = np.argmax(predt, axis = 1)
	predict = predict.astype("int")

	results = np.vstack((index, predict))
	results = np.transpose(results)
	results = pd.DataFrame(results)
	results.to_csv(outputfile, header = ["id", "label"], index = None)