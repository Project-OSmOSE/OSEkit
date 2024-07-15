import numpy as np
import torch
from torch import nn, tensor, utils, device, cuda, optim, long, save
from torch.autograd import Variable


################################################################################################################
#  RUN FUNCTIONS
################################################################################################################

def train_rnn(model, rnn, dataloader, test_loader, criterion, optimizer, num_epochs):
	accuracy = []
	for epoch in range(num_epochs):
		acc_batch = []
		for batch in dataloader:
			optimizer.zero_grad()
			data, labels = batch
			outputs = model(data)
			loss = criterion(outputs.squeeze(dim = 1), labels)
			loss.backward()
			optimizer.step()

			outputs, labels = outputs.squeeze(dim = 1).detach().numpy(), labels.numpy()
			acc_batch.append(np.sum(abs(outputs-labels) < 1)/len(labels))

		accuracy.append(np.mean(acc_batch))
		if (epoch+1) % 1 == 0:
			test_rnn(model, rnn, test_loader, epoch)
			model.train()
		torch.save(model.state_dict(), 'current_training')

def test_rnn(model, rnn, dataloader, epoch):
	model.eval()
	acc_test = []
	all_preds, all_labels = [], []
	with torch.no_grad():
		for batch in dataloader:
			data, labels = batch	
			outputs = model(data)
			outputs, labels = outputs.squeeze().detach().numpy(), labels.numpy()
			all_preds.extend(outputs)
			all_labels.extend(labels)
			acc_test.append(np.sum(abs(outputs-labels) < 1)/len(labels))	
		np.save(os.path.join(os.path.expanduser('~'), str(seq_length) + '_wind_LSTM'), np.vstack((all_preds, all_labels)))



#################################################################################################################
# DATALOADERS
#################################################################################################################

class Wind_Speed(data.Dataset):

	'''
	Dataloader to feed sequence length of chosen frequency and associated ground truth wind speed to a LSTM model.
	Parameters :
		df (DataFrame) : dataframe containing wind speed ground truth data and noise level data.
		frequency (int, float) : Frequency for which data should be fetched
		ground_truth (str): Column name from auxiliary data that stores wind speed data
		sequence_length (int) : Sequence length of consecutive noise level that are fed to LSTM.
	'''
	
	def __init__(self, df, frequency, ground_truth, seq_length):
		self.df = df
		self.seq_length = seq_length
		self.frequency = self.frequency
		self.ground_truth = self.ground_truth

	def __len__(self):
		return self.df.shape[0]
	
	def __getitem__(self, idx):
		spl = torch.FloatTensor(self.df[self.frequency].loc[idx-self.seq_length+1:idx].to_numpy())
		if len(spl) == self.seq_length:
			return spl, torch.tensor(self.df[self.ground_truth].loc[idx], dtype=torch.float)
		else:
			spl = torch.cat((torch.zeros(self.seq_length-len(spl), spl.size(1)), spl))
			return spl, torch.tensor(self.df[self.ground_truth].loc[idx], dtype=torch.float)




##################################################################################################################
#  MODELS
##################################################################################################################

class RNNModel(nn.Module):
	'''
	RNN module that can implement a LSTM
	Number of hiddend im and layer dim are parameters of the model
	'''
	def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
		super(RNNModel, self).__init__()

		# Number of hidden dimensions
		self.hidden_dim = hidden_dim

		# Number of hidden layers
		self.layer_dim = layer_dim

		# RNN
		#self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
		self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

		self.fc1 = nn.Linear(hidden_dim, 128)

		# Readout layer
		self.fc = nn.Linear(128, output_dim)

	def forward(self, x):

		# Initialize hidden state with zeros
		h0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
		c0 = Variable(torch.zeros(self.layer_dim, x.size(0), self.hidden_dim))
		#pdb.set_trace()
		out, (hn, cn) = self.rnn(x, (h0, c0))
		out = self.fc1(out[:, -1, :])
		out = self.fc(out)

		return out
