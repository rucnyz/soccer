# -*- coding: utf-8 -*-
# @Time    : 2022/4/4 20:26
# @Author  : nieyuzhou
# @File    : dense.py
# @Software: PyCharm
from torch import nn


class FeedForward(nn.Module):
    def __init__(self, args):
        super(FeedForward, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(args.feature_dims, args.feature_dims * 2),
            nn.ReLU(),
            nn.Linear(args.feature_dims * 2, args.feature_dims),
            nn.ReLU(),
            nn.Linear(args.feature_dims, args.feature_dims))

    def forward(self, x):
        output = self.network(x)
        return output
