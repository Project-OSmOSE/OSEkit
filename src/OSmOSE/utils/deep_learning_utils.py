import torch
from torch import nn, tensor, utils, device, cuda, optim, long, save
from torch.utils import data
from torch.autograd import Variable
import os
import numpy as np
from tqdm import tqdm

################################################################################################################
#  RUN FUNCTIONS
################################################################################################################


def train_rnn(model, dataloader, test_loader, criterion, optimizer, num_epochs, device):
    accuracy = []
    low_acc = 0
    estimation = np.nan
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        acc_batch = []
        for batch in dataloader:
            optimizer.zero_grad()
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            loss = criterion(outputs.squeeze(dim=1), labels)
            loss.backward()
            optimizer.step()

            # outputs = outputs.squeeze(dim = 1).detach().cpu().numpy()
            # labels = labels.detach().cpu().numpy()
            acc_batch.append(
                (
                    torch.sum(abs(outputs.squeeze(dim=1) - labels) < 1) / labels.size(0)
                ).item()
            )

        accuracy.append(np.mean(acc_batch))
        if (epoch + 1) % 1 == 0:
            low_acc, estimation = test_rnn(
                model, test_loader, epoch, device, low_acc, estimation
            )
            pbar.set_description(f"Evaluation accuracy : {low_acc}")
            model.train()
        torch.save(model.state_dict(), f"model_{epoch}")
    return estimation


def test_rnn(model, dataloader, epoch, device, low_acc, estimation):
    model.eval()
    acc_test = []
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            data, labels = batch
            data = data.to(device)
            labels = labels.to(device)
            outputs = model(data)
            outputs = outputs.squeeze(dim=1)
            acc_test.append(
                (torch.sum(abs(outputs - labels) < 1) / labels.size(0)).item()
            )
            all_preds.extend(outputs.cpu().numpy())
        if np.mean(acc_test) > low_acc:
            return np.mean(acc_test), np.array(all_preds)
        else:
            return low_acc, estimation


#################################################################################################################
# DATALOADERS
#################################################################################################################


class Wind_Speed(data.Dataset):
    """
    Dataloader to feed sequence length of chosen frequency and associated ground truth wind speed to a LSTM model.
    Parameters :
            df (DataFrame) : dataframe containing wind speed ground truth data and noise level data.
            frequency (int, float) : Frequency for which data should be fetched
            ground_truth (str): Column name from auxiliary data that stores wind speed data
            sequence_length (int) : Sequence length of consecutive noise level that are fed to LSTM.
    """

    def __init__(self, df, frequency, ground_truth, seq_length):
        self.df = df.reset_index()
        self.seq_length = seq_length
        self.frequency = frequency
        self.ground_truth = ground_truth

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        spl = self.df[self.frequency].loc[idx - self.seq_length + 1 : idx].to_numpy()
        spl = torch.FloatTensor(spl.reshape(len(spl), -1))
        if len(spl) == self.seq_length:
            return spl, torch.tensor(
                self.df[self.ground_truth].iloc[idx], dtype=torch.float
            )
        else:
            spl = torch.cat((torch.zeros(self.seq_length - len(spl), spl.size(1)), spl))
            return spl, torch.tensor(
                self.df[self.ground_truth].iloc[idx], dtype=torch.float
            )


##################################################################################################################
#  MODELS
##################################################################################################################


class RNNModel(nn.Module):
    """
    RNN module that can implement a LSTM
    Number of hiddend im and layer dim are parameters of the model
    """

    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(RNNModel, self).__init__()

        # Number of hidden dimensions
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.layer_dim = layer_dim

        # RNN
        # self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='relu')
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc1 = nn.Linear(hidden_dim, 128)

        # Readout layer
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):

        # Initialize hidden state with zeros
        h0 = Variable(
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device)
        )
        c0 = Variable(
            torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=x.device)
        )
        # pdb.set_trace()
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc1(out[:, -1, :])
        out = self.fc(out)

        return out
