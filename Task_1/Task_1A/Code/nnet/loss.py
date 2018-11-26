
# NOTE: You can only use Tensor API of PyTorch

import torch

# Extra TODO: Document with proper docstring
def cross_entropy_loss(outputs, labels):
    """Calculates cross entropy loss given outputs and actual labels

    """   
    m = outputs.size()[0]
    creloss = torch.mean(-torch.log(outputs[range(m), labels]))
    return creloss.item()   # should return float not tensor

# Extra TODO: Document with proper docstring
def delta_cross_entropy_softmax(outputs, labels):
    """Calculates derivative of cross entropy loss (C) w.r.t. weighted sum of inputs (Z). 
    
    """
    m = outputs.size()[0]
    outputs[range(m),labels] -= 1
    avg_grads = outputs
    return avg_grads

if __name__ == "__main__":
    pass