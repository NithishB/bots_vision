
# NOTE: You can only use Tensor API of PyTorch

import torch

# Extra TODO: Document with proper docstring
def mbgd(weights, biases, dw1, db1, dw2, db2, dw3, db3, lr):
    """Mini-batch gradient descent
    """
    weights['w1'] += dw1
    weights['w2'] += dw2
    weights['w3'] += dw3
    biases['b1'] += db1
    biases['b2'] += db2
    biases['b3'] += db3

    return weights, biases

if __name__ == "__main__":
    pass