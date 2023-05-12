
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import copy
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import f1_score, confusion_matrix, roc_curve, auc

import warnings
data = []

path_ = 'Users/himamura6/Documents/GitHub/Wearing-a-mask-or-not' 

for num in os.listdir(path_):
    for img_path in os.listdir(f'{path_}/{num}')[:100]:
        data.append([f'{path_}/{num}/{img_path}', img_path, num, num])

df = pd.DataFrame(data, columns=['path', 'filename','category', 'label'])

df.head()

plt.imshow(img_transformed.numpy().transpose((1, 2, 0)))
plt.show()
class Wearing_Dataset_val(torch.utils.data.Dataset):
   def __init__(self, df, features, labels):
        self.features_values = df[features].values
        self.labels = df[labels].values