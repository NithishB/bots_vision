import torch.nn as nn
import torch.nn.functional as F

class FNet(nn.Module):
    # Extra TODO: Comment the code with docstrings
    """Fruit Net

    """
    def __init__(self):
        super(FNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bnorm1 = nn.BatchNorm2d(6)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bnorm2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 5)
        self.fc1 = nn.Linear(16 * 9 * 9, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.bnorm1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.bnorm2(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 9 * 9)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    net = FNet()
