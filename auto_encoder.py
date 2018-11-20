import torch
import torch.nn as nn
import torch.nn.functional as F
from utility import batch_generator


class SingleAutoEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        super(SingleAutoEncoder, self).__init__()
        self.reduce = nn.Linear(input_size, output_size)
        self.restore = nn.Linear(output_size, input_size)

    def forward(self, x):
        x = self.reduce(x)
        return F.softmax(x, 1)

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
                output = self.restore(self.forward(x_set))
                loss = loss_fn(output, x_set)
                loss /= N
                epoch_loss += float(loss)
                loss.backward()
                op.step()
            if verbose:
                print(epoch_num, epoch_loss)
        return self
