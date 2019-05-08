import torch.utils.data as data
from PIL import Image, ImageFile
import os
import json
import random
import numpy as np

ImageFile.LOAD_TRUNCATED_IAMGES = True

# https://github.com/pytorch/vision/issues/81
def PIL_loader(path):
    try:
        with open(path, 'rb') as f:
            return Image.open(f).convert('RGB')
    except IOError:
        print('Cannot load image ' + path)

def get_double_imglist(train_list,train_label):
    list = []
    for idx,name in enumerate(train_list):
        list.append((name[0],name[1],train_label[idx],idx))
    return list

# --------------------------------------A-softmax----------------------------------------

def default_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList

class ImageList(data.Dataset):
    def __init__(self, root, fileList, transform=None, list_reader=default_reader, loader=PIL_loader):
        self.root = root
        self.imgList = list_reader(fileList)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.imgList)

# --------------------------------------Triplet dataset----------------------------------------
def get_triplet_imglist(train_root):
    with open(train_root) as json_file:
        data = json.load(json_file)
    train_lists = data["image_names"]
    train_labels = data["image_labels"]
    return train_lists,train_labels

class TripletList(data.Dataset):
    def __init__(self,root,train_list,phase="train",transform=None,loader=PIL_loader):
        self.root = root
        self.imgList,self.label = get_triplet_imglist(train_list)
        self.nids = int(len(self.label) /2)
        self.transform = transform
        self.loader = loader
        self.phase = phase
    def __getitem__(self, index):
        if self.phase == "train":
            seed = random.randrange(4294967295)
            np.random.seed(seed=seed)
            pair_ids = np.random.choice(self.nids, 2, replace=False)
            # print("pair_ids",pair_ids)
            mask_pos = np.where(self.label == pair_ids[0])[0]
            # print("mask_pos",mask_pos)
            idx_a, idx_p = np.random.choice(mask_pos, 2, replace=False)
            # print("idx_a,idx_p",idx_a,idx_p)
            mask_neg = np.where(self.label == pair_ids[1])[0]
            idx_n = np.random.choice(mask_neg)
            # print("self.root",self.root)
            # print("self.imgList",self.imgList[idx_a])
            data_a = self.loader(os.path.join(self.root,self.imgList[idx_a]))
            data_p = self.loader(os.path.join(self.root, self.imgList[idx_p]))
            data_n = self.loader(os.path.join(self.root, self.imgList[idx_n]))
            if self.transform is not None:
                data_a = self.transform(data_a)
                data_p = self.transform(data_p)
                data_n = self.transform(data_n)
            return data_a,data_p,data_n
        elif self.phase == "test":
            img1 = self.loader(os.path.join(self.root,self.imgList[index][0]))
            img2 = self.loader(os.path.join(self.root,self.imgList[index][1]))
            # print("self.imgList",self.imgList)
            if self.transform is not None:
                img1 = self.transform(img1)
                img2 = self.transform(img2)
            # print("self.label[index]",self.label[index])
            return img1,img2,int(self.label[index][0])

    def __len__(self):
        return len(self.imgList)


# --------------------------------------Random Prototype dataset----------------------------------------
class PImageList(data.Dataset):
    def __init__(self, root, train_root,train_label, transform=None, loader=PIL_loader):
        self.root = root
        self.imgList = get_double_imglist(train_root,train_label)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        imgPath1, imgPath2, target_true,target_false = self.imgList[index]
        img1 = self.loader(os.path.join(self.root, imgPath1))
        img2 = self.loader(os.path.join(self.root, imgPath2))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1,img2,target_true,target_false

    def __len__(self):
        return len(self.imgList)

#----------------------------------------Npair loss-----------------------------------------------------
def get_npair_imglist(train_list,train_label):
    list = []
    for idx,name in enumerate(train_list):
        list.append((name[0],name[1],train_label[idx]))
    return list

class NImageList(data.Dataset):
    def __init__(self, root, train_root,train_label, transform=None, loader=PIL_loader):
        self.root = root
        self.imgList = get_npair_imglist(train_root,train_label)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        imgPath1, imgPaht2, target = self.imgList[index]
        img1 = self.loader(os.path.join(self.root, imgPath1))
        img2 = self.loader(os.path.join(self.root, imgPaht2))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, target
    def __len__(self):
        return len(self.imgList)

#-------------------------------------Contrastive dataset---------------------------------------------
def get_contrastive_imglist(train_root):
    with open(train_root) as json_file:
        data = json.load(json_file)
    train_lists = data["image_names"]
    train_labels = data["image_labels"]
    return train_lists,train_labels

class ContrastiveList(data.Dataset):
    def __init__(self,root,train_list,transform=None,loader=PIL_loader):
        self.root = root
        self.imgList,self.label = get_contrastive_imglist(train_list)
        self.nids = int(len(self.label) /2)
        self.transform = transform
        self.loader = loader
    def __getitem__(self, index):
        seed = random.randrange(4294967295)
        np.random.seed(seed=seed)
        pair_ids = np.random.choice(self.nids, 2, replace=False)
        # print("pair_ids",pair_ids)
        mask_pos = np.where(self.label == pair_ids[0])[0]
        # print("mask_pos",mask_pos)
        idx_a, idx_p = np.random.choice(mask_pos, 2, replace=False)
        # print("idx_a,idx_p",idx_a,idx_p)
        mask_neg = np.where(self.label == pair_ids[1])[0]
        idx_n = np.random.choice(mask_neg)
        # print("self.root",self.root)
        # print("self.imgList",self.imgList[idx_a])
        data_a = self.loader(os.path.join(self.root,self.imgList[idx_a]))
        data_p = self.loader(os.path.join(self.root, self.imgList[idx_p]))
        data_n = self.loader(os.path.join(self.root, self.imgList[idx_n]))
        if self.transform is not None:
            data_a = self.transform(data_a)
            data_p = self.transform(data_p)
            data_n = self.transform(data_n)

        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            return data_a,data_p,int(1)
        else:
            return data_a,data_n,int(0)

    def __len__(self):
        return len(self.imgList)
