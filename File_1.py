from sklearn import datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import numpy as np

iris = datasets.load_iris()
#iris_df = pd.DataFrame(iris['data'], columns=iris['feature_names'])

#print(iris_df)

df= pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                 columns= iris['feature_names'] + ['target'])

df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df)
