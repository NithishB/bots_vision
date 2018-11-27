
# NOTE: You can only use Tensor API of PyTorch

import torch

# Extra TODO: Document with proper docstring
def sigmoid(z):
    """Calculates sigmoid values for tensors
        Arguments:
        z (torch.Tensor) - a tensor for which sigmoid has to be applied
        Returns:
        result (torch.Tensor) - sigmoid activated tensor. Size same as input z
    """
    result = 1/(1+torch.exp(-z))
    return result

# Extra TODO: Document with proper docstring
def delta_sigmoid(z):
    """Calculates derivative of sigmoid function
        Arguments:
        z (torch.Tensor) - a tensor for which gradient sigmoid has to be calculated
        Returns:
        result (torch.Tensor) - sigmoid gradient tensor. Size same as input z
    """
    grad_sigmoid = z * (1 - z)
    return grad_sigmoid

# Extra TODO: Document with proper docstring
def softmax(x):
    """Calculates stable softmax (minor difference from normal softmax) values for tensors
        Arguments:
        x (torch.Tensor) - a tensor for which softmax has to be applied
        Returns:
        stable_softmax (torch.Tensor) - softmax activated tensor. Size same as input x
    """
    x = x - torch.max(x)
    num = torch.exp(x)
    den = torch.sum(num)
    stable_softmax = num / den
    return stable_softmax

if __name__ == "__main__":
    pass