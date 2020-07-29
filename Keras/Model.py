import tensorflow.keras as keras
import tensorflow.keras.layers as nn
import tensorflow.keras.applications as applications

class Net(keras.Model):
	def __init__(self):
		super(Net, self).__init__()
		# pretrained model
		self.pretrained = applications."pretrained model"(weights = "imagenet", include_top = False)

		self.conv = nn.Conv2D(32, "kernel_size",
						 activation = "relu",
						 input_shape = "input shape")
		
		self.pool = nn.MaxPooling2D(pool_size = 2)	# usually 2
		self.drop = nn.Dropout(0.2)					# usually 0.2

		self.last = keras.models.Sequential()
		self.last.add(nn.Flatten())
		self.last.add(nn.Dense("channel"     , activation = "relu"))
		self.last.add(nn.Dense("label number", activation = "softmax"))

	def call(self, img, training = False):
		tmp = self.pretrained(img)		# pretrained model (resnet, vgg16)
		tmp = self.conv(tmp)			# basic convolution layer
		tmp = self.pool(tmp)
		tmp = self.drop(tmp, training = training)
		mod = self.last(tmp)			# correspond to resultant labels
		return mod