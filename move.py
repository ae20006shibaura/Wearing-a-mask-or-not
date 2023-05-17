
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import glob

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

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