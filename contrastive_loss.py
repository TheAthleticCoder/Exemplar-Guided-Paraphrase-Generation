# For both the contrastive losses, input will be of the form
#   shape: (batch_size, hidden_size)
# In the case of the style encoder, hidden_size=768
# and in the case of the content encoder, hidden_size=512
import torch
import torch.nn as nn
from icecream import ic

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(5, 5)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        return self.activation(self.linear(x))

def contrastive_loss(a, b):
    # For both inputs a and b, we have the same shape.
    # shape: (batch_size, hidden_size)
    TAU = 0.5

    # When we take the matrix multiplication of a with a.T, we have the product such that
    # prod[i][j] = a[i] \cdot a[j]. We can exploit this.
    prod_aa = torch.matmul(a, a.T)
    prod_ab = torch.matmul(a, b.T)

    # Since this represents all the dot products we need (and some we don't), taking the max out of
    # these, and subtracting them, will give us all negative values. Since these will soon be
    # exponentiated, that means they will then be all lesser than 1. That prevents them from
    # explosively growing.

    sig_aa = torch.sigmoid(prod_aa)
    sig_ab = torch.sigmoid(prod_ab)
    
    eps_aa = torch.div(sig_aa, TAU)
    eps_ab = torch.div(sig_ab, TAU)

    #Exponentiating them 
    exp_aa = torch.exp(eps_aa)
    exp_ab = torch.exp(eps_ab)

    # Now, we add up all the samples to get the total sample sum. Note that torch.diagonal returns a
    # view on the old tensor, so the base memory for exp_aa and exp_ab is modified. This is ok
    # though, since torch.cat has to copy the tensor anyay, so we need not preserve exp_aa and
    # exp_ab after that.
    sum_samples = torch.sum(exp_aa, dim=1) + torch.sum(exp_ab, dim=1)
    identity_samples = torch.diag(exp_aa)
    pos_samples = torch.diag(exp_ab)

    logits = torch.div(sum_samples - identity_samples, pos_samples)
    loss = torch.log(logits)
    return torch.sum(loss)
