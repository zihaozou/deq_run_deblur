import torch
from torch import nn
import torch.autograd as autograd

class jacobinNet(nn.Module):
    """Unfold network models, i.e. (online) PnP/RED"""
    def __init__(self, dnn: nn.Module):
        super(jacobinNet, self).__init__()
        self.dnn = dnn
    def forward(self, x, create_graph, strict):
        J = autograd.functional.jacobian(self.dnn, x, create_graph=create_graph, strict=strict)
        return J
        
 