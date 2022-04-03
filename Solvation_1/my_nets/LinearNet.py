import torch.nn as nn


class LinearNet3(nn.Module):
    """
        Class represent Linear Network with 3 layers.

        Attributes
        ----------
        in_features : int
            Length of input tensor for first linear layer
        flatten : function
            function that flattens a tensor
        linear_relu_stack : nn.Sequential
            Main stack to define a network

        Methods
        -------
        forward(x)
            predicts a delta_G from given x
        """

    def __init__(self, in_features=10):
        super(LinearNet3, self).__init__()
        self.in_features = in_features
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        """predicts a delta_G from given x"""
        x = self.flatten(x)
        G_solv = self.linear_relu_stack(x)
        return G_solv
