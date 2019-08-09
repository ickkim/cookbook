import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

import datareadin

'''
class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()
        self.config = config
        self.linear1 = nn.Linear(config.input_dim, config.hidden_dim1)
        self.linear2 = nn.Linear(config.hidden_dim2, config.output_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # h: # 16*16 516  2048 2048 1024 1024 516  256
        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(h1))
        h2 = self.dropout(h2)
        out = self.linear8(h2)
        return out
'''

# Focal loss
class Net(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(Net, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

        #self.L1 = nn.Linear(16 * 16, 500)
        #self.L2 = nn.Linear(500, 1024)
        self.config = config
        self.linear1 = nn.Linear(config.input_dim, config.hidden_dim1)
        self.linear2 = nn.Linear(config.hidden_dim2, config.hidden_dim3)
        self.dropout = nn.Dropout(config.dropout)
        self.linear3 = nn.Linear(config.hidden_dim3, config.output_dim)

    def forward(self, inputs, targets):

        input = input.view(batch_size, -1)
        #layer1 = self.L1(input)
        #layer2 = self.L2(layer1)

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return xcvxzc

        h1 = F.relu(self.linear1(x))
        h2 = F.relu(self.linear2(h1))
        h2 = self.dropout(h2)
        out = self.linear3(h2)
        return out




DEVICE = 'cuda:0' #'cuda:0' or 'cpu'
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--input_dim', type=int, default=16*16)
    args.add_argument('--hidden_dim1', type=int, default=516)
    args.add_argument('--hidden_dim2', type=int, default=516)
    args.add_argument('--hidden_dim3', type=int, default=516)

    args.add_argument('--output_dim', type=int, default=1024)
    args.add_argument('--max_epoch', type=int, default=50)
    args.add_argument('--batch_size', type=int, default=20) # number of files(sequences)
    args.add_argument('--initial_lr', type=float, default=0.001)
    args.add_argument('--dropout', type=float, default=0.1)

    ##### params #####
    #input_zise = 16 * 16
    #learning_rate = 0.01
    #training_epoch = 50
    #batch_size = 20
    # num_classes = 2  # 0, 1.... but 1024 per a shot

    config = args.parse_args()
    net = Net(config)

    #criterion = FocalLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



    for epoch in range(2):

        running_loss = 0.0
        for i, data in enumerate(ttt, 0):
            inputs, targets = data

            optimizer.zero_grad()

            outputs = net(inputs)
            running_loss.backward()
            optimizer.step()

            running_loss += loss.item()


