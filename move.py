import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets,transforms

class ResNet_Dataset_val(torch.utils.data.Dataset):
    def__init__(self,transform_RGB=None,transform_GLAY=None):

    RGB_image_names=os.listdir("./used_dataset/val/RGB/")
    depth