import torch.nn as nn
import torchvision
import torchvision.models as models

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.pretrained = models."pretrained model"(pretrained=True)
		self.pretrained = nn.Sequential(*list(self.pretrained.children())[:-2])	# abandon last 2 layers

		self.conv = nn.Sequential(
					nn.Conv2d("input channel", "output channel", "kernel_size", "padding"),
					nn.LeakyReLU(),
					nn.BatchNorm2d("channel"),
					nn.MaxPool2d(2),	# usually 2
					nn.Dropout(0.2),	# usually 0.2
					)

		self.last = nn.Sequential(
					nn.Linear("input channel", "output channel"),
					nn.ReLU(),
					nn.BatchNorm1d("channel"),
					nn.Linear("channel", "label number"),
					)

	def forward(self, img):
		tmp = self.pretrained(img)		# pretrained model (resnet, vgg16)
		tmp = self.conv(tmp)			# basic convolution layer
		tmp = tmp.view(-1, "channel")	# resize to 1 dimension
		mod = self.last(tmp)			# correspond to resultant labels
		return mod
