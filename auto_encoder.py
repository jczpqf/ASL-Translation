import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import batch_generator


class AutoEncoder(nn.Module):
    def __init__(self, encoder_layers):
        super(AutoEncoder, self).__init__()
        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(len(encoder_layers) - 1):
            self.encoder_layers.append(nn.Linear(encoder_layers[i],
                                                 encoder_layers[i + 1]))
        for i in range(len(encoder_layers) - 1, 0, -1):
            self.decoder_layers.append(nn.Linear(encoder_layers[i],
                                                 encoder_layers[i - 1]))

    def forward(self, x):
        for layer in self.encoder_layers:
            x = F.softmax(layer(x), 1)
        return x

    def encode(self, x):
        return self.forward(x)

    def decode(self, x):
        for layer in self.decoder_layers[:-1]:
            x = F.softmax(layer(x), 1)
        return self.decoder_layers[-1](x)

    def train(self, X, batch_size, epochs, op=None, loss_fn=None,
              verbose=True):
        """
        Trains the model to create lower dimensional latent features and
        then uses the latent features to create the orginal features
        """
        if op is None:
            op = torch.optim.Adam(self.parameters())
        if loss_fn is None:
            loss_fn = nn.MSELoss(reduction='sum')
        N = len(X)
        for epoch_num in range(epochs):
            epoch_loss = 0
            training = batch_generator(X, batch_size, shuffle=True)
            for x_set in training:
                x_set = x_set[0]
                op.zero_grad()
                output = self.decode(self.forward(x_set))
                loss = loss_fn(output, x_set)
                loss /= N
                epoch_loss += float(loss)
                loss.backward()
                op.step()
            if verbose:
                print(epoch_num, epoch_loss)
        return self

    def train_greedy(self, X, batch_size, epochs, loss_fn=None, verbose=None):
        if loss_fn is None:
            loss_fn = nn.MSELoss(reduction='sum')
        N = len(X)
        num_layers = len(self.encoder_layers)
        for i in range(num_layers):
            params = [param for param in
                      self.encoder_layers[i].parameters()] + \
                     [param for param in
                      self.decoder_layers[num_layers - i - 1].parameters()]
            op = torch.optim.Adam(params=params)
            for epoch_num in range(epochs):
                epoch_loss = 0
                training = batch_generator(X, batch_size, shuffle=True)
                for x_set in training:
                    x_set = x_set[0]
                    ouput = x_set
                    op.zero_grad()
                    for layer in self.encoder_layers[:i + 1]:
                        ouput = F.softmax(layer(ouput), 1)
                    for layer in self.decoder_layers[num_layers - i - 1:-1]:
                        ouput = F.softmax(layer(ouput), 1)
                    ouput = self.decoder_layers[-1](ouput)
                    loss = loss_fn(ouput, x_set)
                    loss /= N
                    epoch_loss += float(loss)
                    loss.backward()
                    op.step()
                if verbose:
                    print(i, epoch_num, epoch_loss)
