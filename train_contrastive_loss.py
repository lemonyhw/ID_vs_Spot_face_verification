from __future__ import print_function
from __future__ import division
import argparse
import os
import time
import torch
import torch.utils.data
import torch.optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from utils import net_triplet
from utils.dataset import ContrastiveList
from utils.evaluate_prototype import eval
from utils.layer import ContrastiveLoss
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CosFace')

# DATA
parser.add_argument('--train_root', type=str, default='/home3/yhw_datasets/face_recognition/NJN_crop/train',help='path to root path of images')
parser.add_argument('--eval_root',type=str, default='/home3/yhw_datasets/face_recognition/NJN_crop/test',help='path to root path of images')
parser.add_argument('--database', type=str, default='NJN',help='Which Database for train. (NJN)')
parser.add_argument('--train_list', type=str, default="./Data/NJN_triplet_train.json",help='path to training list')
parser.add_argument('--eval_list', type=str, default="./Data/NJN_Random_prototype_test.json",help='path to training list')
parser.add_argument('--batch_size', type=int, default=512,help='input batch size for training (default: 512)')
# Network
parser.add_argument('--network', type=str, default='sphere64',help='Which network for train. (sphere20, sphere64, LResNet50E_IR)')
#pretrain model
parser.add_argument('--pre_model', type=str,default="./models/checkpoint_msra_AL_2/14_1.pth",
                    help='Which network for train. (sphere20, sphere64, LResNet50E_IR)')

# Classifier
parser.add_argument('--num_class', type=int, default=None,help='number of people(class)')
# LR policy
parser.add_argument('--epochs', type=int, default=30,help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.00001,help='learning rate (default: 0.1)')
parser.add_argument('--step_size', type=list, default=None,help='lr decay step')  # [15000, 22000, 26000][80000,120000,140000][100000, 140000, 160000]
parser.add_argument('--momentum', type=float, default=0.9,help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4,metavar='W', help='weight decay (default: 0.0005)')
# Common settings
parser.add_argument('--log_interval', type=int, default=2,help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='./models/checkpoint_NJN_Contrastiveloss/',help='path to save checkpoint')
parser.add_argument('--no_cuda', type=bool, default=False,help='disables CUDA training')
parser.add_argument('--workers', type=int, default=12,help='how many workers to load data')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
args.step_size = [5000,12000,18000]

def main():
    # --------------------------------------model----------------------------------------
    if args.network is 'sphere20':
        model = net_triplet.sphere(type=20)
        model_eval = net_triplet.sphere(type=20)
    elif args.network is 'sphere64':
        model = net_triplet.sphere(type=64)
        model_eval = net_triplet.sphere(type=64)
    elif args.network is 'LResNet50E_IR':
        model = net_triplet.LResNet50E_IR()
        model_eval = net_triplet.LResNet50E_IR()
    else:
        raise ValueError("NOT SUPPORT NETWORK! ")

    model = torch.nn.DataParallel(model).to(device)
    model_eval = torch.nn.DataParallel(model_eval).to(device)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # ------------------------------------load image---------------------------------------
    #train set
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(128),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    train_loader = torch.utils.data.DataLoader(ContrastiveList(
        root=args.train_root, train_list=args.train_list,transform=train_transform),
        batch_size=args.batch_size, shuffle=True)
    # --------------------------------loss function and optimizer-----------------------------
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    criterion = ContrastiveLoss()
    # --------------------------------Load pretrained model parameters-----------------------------
    if args.pre_model:
        checkpoint = torch.load(args.pre_model)
        model.module.load_state_dict(checkpoint)
    # --------------------------------Train-----------------------------
    print("start training")
    for epoch in range(1,args.epochs + 1):
        train(train_loader, model,optimizer,criterion,epoch)
        model_name = args.save_path + str(epoch) + '.pth'
        model.module.save(model_name)
        eval(model_eval, epoch,
             model_name,
             args.eval_root,
             args.eval_list,
             device,
             batch_size=400, workers=12)
    print('Finished Training')

def train(train_loader,model,optimizer,criterion,epoch):
    model.train()
    print_with_time('Epoch {} start training'.format(epoch))
    time_curr = time.time()
    loss_display = 0.0
    for batch_idx, (data_a,data_p,target) in enumerate(train_loader):
        iteration = (epoch - 1) * int(len(train_loader) / 2) + batch_idx
       # adjust_learning_rate(optimizer, iteration, args.step_size)
        data_a,data_p,target = data_a.to(device), data_p.to(device),target.to(device)
        out_a, out_p = model(data_a),model(data_p)
        loss = criterion(out_a,out_p,target)
        loss_display += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and batch_idx>0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            print_with_time('Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.6f}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, batch_idx * len(data_a), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    iteration, loss_display, time_used, args.log_interval))
            time_curr = time.time()
            loss_display = 0.0

def print_with_time(string):
    print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()) + string)

def adjust_learning_rate(optimizer, iteration, step_size):
    """Sets the learning rate to the initial LR decayed by 10 each step size"""
    if iteration in step_size:
        lr = args.lr * (0.1 ** (step_size.index(iteration) + 1))
        print_with_time('Adjust learning rate to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        pass

if __name__ == '__main__':
    print(args)
    main()
