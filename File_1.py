from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()       # fc stands for fully connected
        self.fc1 = nn.Linear(4,10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,3)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = self.fc2(X)
        X = self.fc3(X)
        X = self.softmax(X)

        return X

iris = datasets.load_iris()

iris_df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])

iris_df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(iris_df)

net = Net()
print(net)
