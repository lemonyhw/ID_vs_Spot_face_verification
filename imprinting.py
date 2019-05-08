from __future__ import print_function
from __future__ import division
import argparse
import os
import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
# import net_1
from utils.dataset import PImageList
import json
from utils import net_msra
import scipy.io as scio
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'

parser = argparse.ArgumentParser(description='Imprinting')
parser.add_argument('--root_path', type=str, default='/home3/yhw_datasets/face_recognition/NJN_crop/train', help='path to root path of images')
parser.add_argument('--train_list', type=str, default='./Data/NJN_Random_prototype_train_300000.json', help='path to root path of images')
parser.add_argument('--batch_size', type=int, default=600, help='input batch size for training (default: 512)')

parser.add_argument('--network', type=str, default='sphere64', help='Which network for train. (sphere20, sphere64, LResNet50E_IR)')
# parser.add_argument('--pre_model', type=str, default='./models/checkpoint_NJN_tripletloss/4000.pth')
parser.add_argument('--pre_model', type=str, default='./models/checkpoint_NJN_Npairloss/50.pth')
parser.add_argument('--no_cuda', type=bool, default=False, help='disables CUDA training')
parser.add_argument('--workers', type=int, default=12, help='how many workers to load data')
parser.add_argument('--prototype', type=str, default="ID", help='which feature for initializing weights')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

def imprint(novel_loader,model,train_num):
    model.eval()
    new_weight = torch.zeros(train_num, 512)
    with torch.no_grad():
        for batch_idx, (input_id, input_spot, target,_) in enumerate(novel_loader):
            # print(target)
            input_id, input_spot, target = input_id.to(device), input_spot.to(device), target.to(device)
            output_id = model(input_id)
            if args.prototype == "ID":
                output = output_id
            elif args.prototype == "AVG":
                output_spot = model(input_spot)
                output = (output_id + output_spot) / 2  # w = (Wid + Wspot)/2
            for idx in range(args.batch_size):
                tmp = output[idx, :].cpu()
                tmp1 = tmp / tmp.norm(p=2)
                new_weight[target[idx], :] = tmp1
    return new_weight

if args.network is 'sphere20':
    model = net_msra.sphere(type=20)
    model_eval = net_msra.sphere(type=20)
elif args.network is 'sphere64':
    model = net_msra.sphere(type=64)
    model_eval = net_msra.sphere(type=64)
elif args.network is 'LResNet50E_IR':
    model = net_msra.LResNet50E_IR()
    model_eval = net_msra.sphere(type=64)
else:
    raise ValueError("NOT SUPPORT NETWORK! ")

model = torch.nn.DataParallel(model).to(device)

with open(args.train_list) as json_file:
    data = json.load(json_file)
train_list =data["image_names"]
train_label = data["image_labels"]
train_list = train_list[:300000]
train_label = train_label[:300000]
train_num = len(train_list)
print('length of train Database: ' + str(len(train_list)))

#novel set,use [ID-prototype] or [avg-prototype]
novel_transform = transforms.Compose([
    transforms.CenterCrop(128),
    transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
])

novel_loader = torch.utils.data.DataLoader(
    PImageList(root=args.root_path, train_root=train_list,train_label=train_label,transform=novel_transform),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True, drop_last=False)

checkpoint = torch.load(args.pre_model)
model.module.load_state_dict(checkpoint)

new_weight = imprint(novel_loader,model,train_num)
new_weight = new_weight.numpy()
# np.save("Prototype_weight_ID.npy",new_weight)
scio.savemat("./Data/Prototype_weight_ID_300000_npair.mat",{"weight":new_weight,"label":train_label})
print(new_weight.shape)
