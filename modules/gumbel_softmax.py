import torch
import torch.nn as nn
from torch.autograd import Variable

class GumbelSoftmax(nn.Module):
    def __init__(self, use_cuda=False):
        super(GumbelSoftmax, self).__init__()
        self.using_cuda = use_cuda
        self.softmax = nn.Softmax(dim=1)
        self.temp = 1

    def forward(self, x):
        if self.using_cuda:
            U = Variable(torch.rand(x.size()).cuda())
        else:
            U = Variable(torch.rand(x.size()))
        y = x -torch.log(-torch.log(U + 1e-20) + 1e-20)
        return self.softmax(y/self.temp)
