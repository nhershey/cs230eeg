"""Defines the neural networks, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, params):
        """
        We define a network based on the parameter to predict whether the slice is a seizure. The current models are:
        - base: flatten the matrix and apply a neural net with one hidden layer
        - conv: apply a standard vision neural net of 3 convolutional layers (filters, batch norm, pool, relu)
                and then flatten and apply two fully connected layers with batch norm and dropout
        - lstm: a

        Args:
            params: (Params) contains num_channels
        """
        super(Net, self).__init__()
        self.type = params.type

        if self.type == "reg":
            self.fc = nn.Linear(50000,1)

        if self.type == "conv":
            self.num_channels = params.num_channels
            # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
            # stride, padding). We also include batch normalization layers that help stabilise training.
            self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(self.num_channels)
            self.conv2 = nn.Conv2d(self.num_channels, self.num_channels*2, 3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(self.num_channels*2)
            self.conv3 = nn.Conv2d(self.num_channels*2, self.num_channels*4, 3, stride=1, padding=1)
            self.bn3 = nn.BatchNorm2d(self.num_channels*4)

            # 2 fully connected layers to transform the output of the convolution layers to the final output
            self.fc1 = nn.Linear(250*3*self.num_channels*4, self.num_channels*4)
            self.fcbn1 = nn.BatchNorm1d(self.num_channels*4)
            self.fc2 = nn.Linear(self.num_channels*4, 1)
            self.dropout_rate = params.dropout_rate
        
        elif self.type == "base":
            # simple base model
            self.fc_1 = nn.Linear(50000,1000)
            self.fc_2 = nn.Linear(1000,30)
            self.fc_3 = nn.Linear(30,1)

        elif self.type == "lstm":
            # input_size, hidden_size, num_layers
            self.lstm = nn.LSTM(25, 20, 2)
            # the fully connected layer transforms the output to give the final output layer
            self.fc = nn.Linear(20, 1)
            if params.bidirectional == 1:
                self.lstm = nn.LSTM(25, 20, 2, bidirectional=True)
                self.fc = nn.Linear(40, 1)

        if self.type == "deepconv":
            # in_channels, out_channels, kernel_size
            self.conv1 = nn.Conv2d(1, 25, (10,1), stride=1, padding=1)
            self.conv2 = nn.Conv2d(25, 50, (10,1), stride=1, padding=1)
            self.conv3 = nn.Conv2d(50, 100, (10,1), stride=1, padding=1)
            self.conv4 = nn.Conv2d(100, 200, (10,1), stride=1, padding=1)
            self.conv5 = nn.Conv2d(200, 400, (10,1), stride=1, padding=1)
            self.fc1 = nn.Linear(400*4*35, 400)
            self.fcbn1 = nn.BatchNorm1d(400)
            self.fc2 = nn.Linear(400, 1)
            self.dropout_rate = params.dropout_rate

        if self.type == "deepconv_nodo":
            # in_channels, out_channels, kernel_size
            self.conv1 = nn.Conv2d(1, 25, (10,1), stride=1, padding=1)
            self.conv2 = nn.Conv2d(25, 50, (10,1), stride=1, padding=1)
            self.conv3 = nn.Conv2d(50, 100, (10,1), stride=1, padding=1)
            self.conv4 = nn.Conv2d(100, 200, (10,1), stride=1, padding=1)
            self.conv5 = nn.Conv2d(200, 400, (10,1), stride=1, padding=1)
            self.fc1 = nn.Linear(400*4*35, 400)
            self.fcbn1 = nn.BatchNorm1d(400)
            self.fc2 = nn.Linear(400, 1)

        if self.type == "deeplstm":
            self.lstm = nn.LSTM(25, 20, 2, bidirectional=True)
            self.fc1 = nn.Linear(40, 10)
            self.fcbn1 = nn.BatchNorm1d(10)
            self.fc2 = nn.Linear(10, 1)



            

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.
        Args:
            s: (Variable) contains a batch of images, of dimension batch_size x 2000 x 5.

        Returns:
            out: (Variable) dimension batch_size indicating probability of a seizure.

        Note: the dimensions after each step are provided
        """
        if (self.type == "reg"):
            s = s.view(-1, 50000)
            s = self.fc(s)
            return F.sigmoid(s)

        if (self.type == "conv"):
            # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
            s = s.unsqueeze(1)                                  # -> batch_size x 1 x 2000 x 25
            s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 2000 x 25
            s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 1000 x 12
            s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 1000 x 12
            s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 500 x 6
            s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 500 x 6
            s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 250 x 3

            # flatten the output for each image
            s = s.view(-1, 250*3*self.num_channels*4)             # batch_size x 8*8*num_channels*4
            # apply 2 fully connected layers with dropout
            s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
                p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
            s = self.fc2(s)                                     # batch_size x 6
            return F.sigmoid(s)

        elif (self.type == "base"):
            s = s.view(-1, 50000) 
            s = F.relu(self.fc_1(s))
            s = F.relu(self.fc_2(s))
            s = self.fc_3(s)
            return F.sigmoid(s)

        elif (self.type == "lstm"):
            s = s.transpose(0, 1)
            # Forward propagate RNN
            out, _ = self.lstm(s)  
            # Decode hidden state of last time step (seq_len, batch, hidden_size * num_directions)
            last_out = out[-1,:,:]
            out = self.fc(last_out)  
            return F.sigmoid(out)

        elif (self.type == "deeplstm"):
            s = s.transpose(0, 1)
            # Forward propagate RNN
            out, hidden = self.lstm(s)  
            # Decode hidden state of last time step (seq_len, batch, hidden_size * num_directions)
            s = F.relu(F.max_pool1d(hidden[0], 2))
            s = s.transpose(0, 1)
            s = s.contiguous()
            s = s.view(-1, 4 * 10)
            s = F.relu(self.fcbn1(self.fc1(s)))
            s = self.fc2(s)  
            return F.sigmoid(s)

        elif (self.type == "deepconv"):
            s = s.unsqueeze(1)
            s = F.elu(self.conv1(s))
            s = F.max_pool2d(s, (3, 1))
            s = F.elu(self.conv2(s))
            s = F.max_pool2d(s, (3, 1))
            s = F.elu(self.conv3(s))
            s = F.max_pool2d(s, (3, 1))
            s = F.elu(self.conv4(s))
            s = F.max_pool2d(s, (3, 1))
            s = F.elu(self.conv5(s))
            s = F.max_pool2d(s, (3, 1))
            s = s.contiguous()
            s = s.view(-1, 400*4*35)
            s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),p=self.dropout_rate, training=self.training)
            s = self.fc2(s)
            return F.sigmoid(s)

        elif (self.type == "deepconv_nodo"):
            s = s.unsqueeze(1)
            s = F.relu(self.conv1(s))
            s = F.max_pool2d(s, (3, 1))
            s = F.relu(self.conv2(s))
            s = F.max_pool2d(s, (3, 1))
            s = F.relu(self.conv3(s))
            s = F.max_pool2d(s, (3, 1))
            s = F.relu(self.conv4(s))
            s = F.max_pool2d(s, (3, 1))
            s = F.relu(self.conv5(s))
            s = F.max_pool2d(s, (3, 1))
            s = s.contiguous()
            s = s.view(-1, 400*4*35)
            s = F.relu(self.fcbn1(self.fc1(s)))
            s = self.fc2(s)
            return F.sigmoid(s)


def loss_fn(outputs, labels):
    """
    Compute the cross entropy loss given outputs and labels.

    Args:
        outputs: (Variable) dimension batch_size - output of the model
        labels: (Variable) dimension batch_size, where each element is 0 or 1

    Returns:
        loss (Variable): cross entropy loss for all slices in the batch
    """
    # ((1-labels.float()) * torch.log(1 - outputs) + labels.float() * torch.log(outputs)).mean()
    loss = nn.BCELoss()
    outputs = outputs.squeeze()
    out = loss(outputs, labels.float())
    return out


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all slices.

    Args:
        outputs: (np.ndarray) dimension batch_size - sigmoid output of the model
        labels: (np.ndarray) dimension batch_size - each element is 0 (nonseizure) or 1 (seizure)

    Returns: (float) accuracy in [0,1]
    """
    outputs_round = np.rint(outputs)
    labels = labels.reshape(outputs_round.shape)
    accuracy = np.sum(outputs_round==labels)/float(labels.size)
    # import ipdb; ipdb.set_trace()
    return accuracy

def f1score(outputs, labels):
    """
    Compute the f1 score, given the outputs and labels for all slices.

    Args:
        outputs: (np.ndarray) dimension batch_size - sigmoid output of the model
        labels: (np.ndarray) dimension batch_size - each element is 0 (nonseizure) or 1 (seizure)

    Returns: (float) f1 score in [0,1]
    """
    outputs_round = np.rint(outputs)
    labels = labels.reshape(outputs_round.shape)
    numerator = np.sum(np.logical_and(outputs_round == 1, outputs_round == labels))
    precision_denom = max(outputs_round.sum(), 1e-8)
    recall_denom = max(labels.sum(), 1e-8)
    precision = 1.0 * numerator / precision_denom
    recall = 1.0 * numerator / recall_denom
    if (precision + recall == 0):
        return 0
    f1 = 2 * (precision * recall) / (precision + recall)
    # import ipdb; ipdb.set_trace()
    return f1


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'f1': f1score,
}
