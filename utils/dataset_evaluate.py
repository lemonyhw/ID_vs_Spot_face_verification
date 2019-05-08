import torch.utils.data as data
from PIL import Image, ImageFile
import os
import json
import random
import numpy as np
from torchvision.transforms import functional as F


def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def get_double_imglist(train_list,):
    list = []
    for idx,name in enumerate(train_list):
        # print("idx",idx)
        list.append((name[0],name[1]))
    return list

class PImageList(data.Dataset):
    def __init__(self, root, train_root,transform=None, loader=PIL_loader):
        self.root = root
        self.imgList = get_double_imglist(train_root)
        self.transform = transform
        self.loader = loader
    def __getitem__(self, index):
        imgPath1, imgPath2 = self.imgList[index]
        img_s = self.loader(os.path.join(self.root, imgPath1))
        img_I = self.loader(os.path.join(self.root, imgPath2))

        if self.transform is not None:
            img1 = self.transform(img_s)
            img1_ = self.transform(F.hflip(img_s))
            img2 = self.transform(img_I)
            img2_ = self.transform(F.hflip(img_I))
            return img1,img2,img1_,img2_
        else:
            img1_ = F.hflip(img_s)
            img2_ = F.hflip(img_I)
            return img_s,img_I,img1_,img2_
    def __len__(self):
        return len(self.imgList)