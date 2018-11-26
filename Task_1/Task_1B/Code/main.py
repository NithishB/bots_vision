
# Homecoming (eYRC-2018): Task 1B
# Fruit Classification with a CNN

from model import FNet
from utils.dataset import create_and_load_meta_csv_df, ImageDataset
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim


def train_model(dataset_path, debug=False, destination_path='./meta', save=False):
	"""Trains model with set hyper-parameters and provide an option to save the model.

	This function should contain necessary logic to load fruits dataset and train a CNN model on it. It should accept dataset_path which will be path to the dataset directory. You should also specify an option to save the trained model with all parameters. If debug option is specified, it'll print loss and accuracy for all iterations. Returns loss and accuracy for both train and validation sets.

	Args:
		dataset_path (str): Path to the dataset folder. For example, '../Data/fruits/'.
		debug (bool, optional): Prints train, validation loss and accuracy for every iteration. Defaults to False.
		destination_path (str, optional): Destination to save the model file. Defaults to ''.
		save (bool, optional): Saves model if True. Defaults to False.

	Returns:
		loss (torch.tensor): Train loss and validation loss.
		accuracy (torch.tensor): Train accuracy and validation accuracy.
	"""
	df, traindf, testdf = create_and_load_meta_csv_df(dataset_path, destination_path, True, 0.7)
	traindataset = ImageDataset(traindf)
	testdataset = ImageDataset(testdf)

	trainloader = DataLoader(traindataset, batch_size=32, shuffle=True, num_workers=2)
	testloader = DataLoader(testdataset, batch_size=32, shuffle=True, num_workers=2) 

	net = FNet()
	
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	for epoch in range(2):  # loop over the dataset multiple times

	    running_loss = 0.0
	    for i, data in enumerate(trainloader, 0):
		# get the inputs
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
		    print('[%d, %5d] loss: %.3f' %
		          (epoch + 1, i + 1, running_loss / 2000))
		    running_loss = 0.0

	print('Finished Training')

if __name__ == "__main__":
	train_model('../../Data/fruits/', save=True, destination_path='./')
