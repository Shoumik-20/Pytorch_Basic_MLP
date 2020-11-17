from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
#print(iris_df)

train_X, test_X, train_y, test_y = train_test_split(iris_df[iris_df.columns[0:4]].values, iris_df.target.values, test_size=0.25)

train_X = Variable(torch.Tensor(train_X).float())
test_X = Variable(torch.Tensor(test_X).float())
train_y = Variable(torch.Tensor(train_y).long())
test_y = Variable(torch.Tensor(test_y).long())

net = Net()

loss_function = nn.CrossEntropyLoss()     # loss function
optimizer = torch.optim.Adam(net.parameters(), lr = 0.01)   # Adam optimizer is used

for epoch in range(3000):
    net.zero_grad()        #set gradient to zero
    out = net(train_X)     # output calculated by neural network
    loss = loss_function( out, train_y)   # loss between output calculated and  expected output
    loss.backward()
    optimizer.step()    #optimize the weights
    print(loss)


print(net)
