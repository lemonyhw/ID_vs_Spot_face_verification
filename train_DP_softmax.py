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
from utils import net, layer
from utils.dataset import PImageList
import scipy.io as sio
import json

os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3,4'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CosFace')

# DATA
parser.add_argument('--root_path', type=str, default='/home3/yhw_datasets/face_recognition/NJN_ID_SPOT_crop/',help='path to root path of images')
parser.add_argument('--database', type=str, default='NJN',help='Which Database for train. (NJN)')
parser.add_argument('--train_list', type=str, default=None,help='path to training list')
parser.add_argument('--batch_size', type=int, default=512,help='input batch size for training (default: 512)')
# Network
parser.add_argument('--network', type=str, default='sphere64',help='Which network for train. (sphere20, sphere64, LResNet50E_IR)')
# Classifier
parser.add_argument('--num_class', type=int, default=100000,help='number of people(class)')
parser.add_argument('--classifier_type', type=str, default='RP',help='Which classifier for train. (MCP, AL, L,ARC,TRIP,RP,DP)')
#Extract feature
parser.add_argument('--prototype', type=str, default="ID",help='which feature for initializing weights')
# LR policy
parser.add_argument('--epochs', type=int, default=30,help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.1,help='learning rate (default: 0.1)')
parser.add_argument('--step_size', type=list, default=None,help='lr decay step')  # [15000, 22000, 26000][80000,120000,140000][100000, 140000, 160000]
parser.add_argument('--momentum', type=float, default=0.9,help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4,metavar='W', help='weight decay (default: 0.0005)')
# Common settings
parser.add_argument('--log_interval', type=int, default=100,help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='checkpoint-sphereface64_TripletLoss/',help='path to save checkpoint')
parser.add_argument('--no_cuda', type=bool, default=False,help='disables CUDA training')
parser.add_argument('--workers', type=int, default=12,help='how many workers to load data')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.database is "NJN":
    train_filename = sorted(os.listdir(args.root_path))
    args.step_size =[]
else:
    raise ValueError("NOT SUPPORT DATABASE! ")

def main():
    # --------------------------------------model----------------------------------------
    if args.network is 'sphere20':
        model = net.sphere(type=20)
        model_eval = net.sphere(type=20)
    elif args.network is 'sphere64':
        model = net.sphere(type=64)
        model_eval = net.sphere(type=64)
    elif args.network is 'LResNet50E_IR':
        model = net.LResNet50E_IR()
        model_eval = net.LResNet50E_IR()
    else:
        raise ValueError("NOT SUPPORT NETWORK! ")

    model = torch.nn.DataParallel(model).to(device)
    model_eval = torch.nn.DataParallel(model_eval).to(device)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # ------------------------------------load image---------------------------------------
    # train set
    with open(args.train_list) as json_file:
        data = json.load(json_file)
    train_list = data["image_names"]
    train_label = data["image_labels"]
    train_num = len(train_list)
    print('length of train Database: ' + str(train_num))
    print('Number of Identities: ' + str(args.num_class))

    # --------------------------------Updated prototype matrix-----------------------------
    checkpoint = torch.load(args.pre_model)
    model.module.load_state_dict(checkpoint)

    data = sio.loadmat(args.new_weight)
    new_weight = data["weight"]
    new_weight = torch.from_numpy(new_weight)

    # --------------------------------Queue initialization-----------------------------
    #compute feature distance
    queue_stack = torch.zeros(len(new_weight),len(new_weight))
    for idx1 in range(len(new_weight)):
        for idx2 in range(len(new_weight)):
            if idx1 == idx2:
                queue_stack[idx1,idx2] == 0
            else:
                dist = torch.dist(new_weight[idx1],new_weight[idx2])
                queue_stack[idx1,idx2] == dist

    queue_stack_sort,queue_stack_index = torch.sort(queue_stack, descending=False)

    #candidate
    # indices1 = torch.LongTensor([0, args.candidate])
    # NC300 = torch.index_select(index, 1, indices1)
    # NC300_unique = torch.unique(NC300) #remove the repeat elements

    #queue
    indices2 = torch.LongTensor([0,args.queue_number])
    NCk = torch.inex_select(queue_stack_index, 1, indices2)
    NCk_positive = torch.unique(NCk) # remove the repeat elements

    NCk_all = torch.arange(0,args.num_class)
    mask_ones = torch.ones(args.num_class)
    mask_ones[NCk_positive] = 0
    mask_ones = mask_ones.byte()
    NCk_negative = torch.masked_select(NCk_all,mask_ones)

    if args.queue_number == 10:
        Niter = 300
        positive,negative = gernerate_NCk(NCk_positive,NCk_negative,Niter)
    elif args.queue_number == 20:
        Niter = 600
        positive,negative = gernerate_NCk(NCk_positive,NCk_negative,Niter)
    elif args.queue_number == 50:
        Niter = 1500
        positive,negative = gernerate_NCk(NCk_positive,NCk_negative,Niter)
    elif args.queue_number == 100:
        Niter = 3000
        positive,negative = gernerate_NCk(NCk_positive,NCk_negative,Niter)
    elif args.queue_number == 300:
        Niter = 10000
        positive,negative = gernerate_NCk(NCk_positive,NCk_negative,Niter)

    # --------------------------------loss function and optimizer-----------------------------
    if args.classifier_type != "DUM":
        criterion = torch.nn.CrossEntropyLoss().to(device)
    else:
        criterion = []

    # --------------------------------dataset loader-----------------------------
    list_label = list(zip(train_list, train_label))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(128),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])

    # --------------------------------Train-----------------------------
    for iters in range(args.iterations):
        classifier = {
            'MCP': layer.MarginCosineProduct(512, args.num_class).to(device),
            'AL': layer.AngleLinear(512, args.num_class).to(device),
            'L': torch.nn.Linear(512, args.num_class, bias=False).to(device),
            "ARC": layer.ArcMarginProduct(512, args.num_class, s=30, m=0.4, easy_margin=False).to(device),
            "DUM": layer.DumLoss(512, args.num_classes).to(device),
            "TRIP": layer.TripletLoss().to(device)
        }[args.classifier_type]
        train_loader = torch.utils.data.DataLoader(
            PImageList(root=args.root_path, train_root=sub_train_list, train_label=sub_train_label,
                       transform=train_transform),
            batch_size=int(args.batch_size / 2), shuffle=True,
            num_workers=args.workers, pin_memory=False, drop_last=True)
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': classifier.parameters()}],
                                    lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
        target,update_queue_list = train(train_loader, model, classifier, criterion,optimizer, epoch,positive)

        if iters % 1000 == 0:
            model.module.save(args.save_path + 'DummpyFace_' + str(iters) + '_checkpoint.pth')
        #test(test_loader, model_eval,classifier,criterion,epoch)
    print('Finished Training')

def train(train_loader, model, classifier,criterion, optimizer, epoch,positive):
    print_with_time('Epoch {} start training'.format(epoch))
    time_curr = time.time()
    model.train()
    loss_display = 0.0
    update_queue_list = []
    for batch_idx, (data,target,target1) in enumerate(train_loader):
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        # adjust_learning_rate(optimizer, iteration, args.step_size)
        data, target, target1 = data.to(device), target.to(device),target1.to(device)
        if args.classifier_type != "DUM":
            output = model(data)
            if isinstance(classifier, torch.nn.Linear):
                _,output = classifier(output)
            else:
                _,output = classifier(output, target)
            loss = criterion(output, target)
        else:
            output = model(data,target1)
            probobility,loss = classifier(output,target)
            # mask = torch.zeros(args.batch_size,args.num_class).scatter_(1,target,1)
            # Eergy = torch.masked_select(probobility,mask).sum()
            predicted,pred_index = torch.max(probobility,dim = 1)

            for id in range(len(target1)):
                if pred_index[id] == target1[id]:
                    continue
                else:
                    mask_index = torch.masked_select(positive,positive == pred_index[id])
                    if mask_index.size()[0] == False:
                        continue
                    else:
                        update_queue_list.append(mask_index.numpy())

        loss_display += loss.item()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            if args.classifier_type is 'MCP':
                INFO = ' Margin: {:.4f}, Scale: {:.2f}'.format(classifier.m, classifier.s)
            elif args.classifier_type is 'AL':
                INFO = ' lambda: {:.4f}'.format(classifier.lamb)
            else:
                INFO = ''
            print_with_time(
                'Train Epoch: {} [{}/{} ({:.0f}%)]{}, Loss: {:.6f}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    iteration, loss_display, time_used, args.log_interval) + INFO)
            time_curr = time.time()
            loss_display = 0.0
    return target1,update_queue_list

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


def gernerate_NCk(NCk_positive, NCk_negative, Niter):
    if len(NCk_positive) <= Niter:
        NCk_positive = torch.cat((NCk_positive, NCk_negative[:Niter - len(NCk_positive)]), 1)
        NCk_negative = NCk_negative[Niter - len(NCk_positive):]
    else:
        NCk_positive = NCk_positive[:Niter]
        NCk_negative = torch.cat((NCk_negative, NCk_positive[Niter:]))
    return NCk_positive, NCk_negative

if __name__ == '__main__':
    print(args)
    main()
