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
from utils import net,net_msra,layer, lfw_eval,evaluate_prototype
from utils.dataset import ImageList

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CosFace')
# DATA
# parser.add_argument('--root_path', type=str, default='/home3/yhw_datasets/face_recognition/CASIA-WebFace_crop_1/',help='path to root path of images')

parser.add_argument('--root_path', type=str, default='/home3/yhw_datasets/face_recognition/msra_crop/',help='path to root path of images')
parser.add_argument('--database', type=str, default='Msra',help='Which Database for train. (WebFace, Msra)')
parser.add_argument('--train_list', type=str, default=None,help='path to training list')
parser.add_argument('--eval_root',type=str, default='/home3/yhw_datasets/face_recognition/NJN_crop/test/',help='path to root path of images')
parser.add_argument('--eval_list', type=str, default="./Data/NJN_Random_prototype_test.json",help='path to training list')
parser.add_argument('--batch_size', type=int, default=384,help='input batch size for training (default: 512)')
parser.add_argument('--pre_model', type=str, default="./models/checkpoint_msra_AL_2/3_1.pth")
# Network
parser.add_argument('--network', type=str, default='sphere64',help='Which network for train. (sphere20, sphere64, LResNet50E_IR)')
# Classifier
parser.add_argument('--num_class', type=int, default=None,help='number of people(class)')
parser.add_argument('--classifier_type', type=str, default='AL',help='Which classifier for train. (MCP, AL, L,ARC)')
# LR policy
parser.add_argument('--start_epoch', type=int, default=15)
parser.add_argument('--epochs', type=int, default=20,help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.01,help='learning rate (default: 0.1)')
parser.add_argument('--step_size', type=list, default=None,
                    help='lr decay step')  # [15000, 22000, 26000][80000,120000,140000][100000, 140000, 160000]
parser.add_argument('--momentum', type=float, default=0.9,help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4,metavar='W', help='weight decay (default: 0.0005)')
# Common settings
parser.add_argument('--log_interval', type=int, default=100,help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='./models/checkpoint_msra_AL_2/',help='path to save checkpoint')
parser.add_argument('--no_cuda', type=bool, default=False,help='disables CUDA training')
parser.add_argument('--workers', type=int, default=12,help='how many workers to load data')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")


if args.database is 'WebFace':
    args.train_list = './Data/webface_10574.txt'
    args.num_class = 10574
    args.step_size = [16000, 24000]
elif args.database is 'Msra':
    args.train_list = './Data/msra.txt'
    args.num_class = 86869
    # args.step_size = [31000,82000,120000]

    args.step_size = [51000,110000,150000]
else:
    raise ValueError("NOT SUPPORT DATABASE! ")


def main():
    # --------------------------------------model----------------------------------------
    if args.network is 'sphere20':
        model = net.sphere(type=20)
        model_eval = net.sphere(type=20)
    elif args.network is 'sphere64':
        pre_model = net.sphere(type=64)
        model = net_msra.sphere(type=64)
        model_eval = net_msra.sphere(type=64)
    elif args.network is 'LResNet50E_IR':
        model = net.LResNet50E_IR()
        model_eval = net.LResNet50E_IR()
    else:
        raise ValueError("NOT SUPPORT NETWORK! ")

    pre_model = torch.nn.DataParallel(pre_model).to(device)

    model = torch.nn.DataParallel(model).to(device)
    model_eval = torch.nn.DataParallel(model_eval).to(device)
    # print(model)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    model.module.save(args.save_path + '0_1.pth')

    # 512 is dimension of feature
    classifier = {
        'MCP': layer.MarginCosineProduct(512, args.num_class).to(device),
        'AL' : layer.AngleLinear(512, args.num_class).to(device),
        'L'  : torch.nn.Linear(512, args.num_class, bias=False).to(device),
        # "ARC": layer.ArcMarginProduct(512, args.num_class, s=30, m=0.4, easy_margin=False).to(device),
    }[args.classifier_type]

    # classifier = torch.nn.DataParallel(classifier).to(device)
    # classifier.save(args.save_path + '0_2.pth')


    # ------------------------------------load image---------------------------------------

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(128),
        # transforms.RandomCrop(128),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    train_loader = torch.utils.data.DataLoader(
        ImageList(root=args.root_path, fileList=args.train_list,
                  transform=train_transform),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    print('length of train Database: ' + str(len(train_loader.dataset)))
    print('Number of Identities: ' + str(args.num_class))

    # --------------------------------loss function and optimizer-----------------------------
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD([{'params': model.parameters(),'lr':args.lr}, {'params': classifier.parameters(),'lr':args.lr*10}],
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    #
    # optimizer = torch.optim.SGD([{'params': filter(lambda  p: p.requires_grad,model.parameters())}, {'params': classifier.parameters()}],
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)

    if args.pre_model:
        # print(args.pre_model)
        checkpoint = torch.load(args.pre_model)
        model.module.load_state_dict(checkpoint)

        # print(pre_model.state_dict())
        # model_dict = model.state_dict()
        # pre_model = {k:v for k,v in pre_model.state_dict().items() if k in model_dict}
        # model_dict.update(pre_model)
        # model.load_state_dict(model_dict)
        # print(model.state_dict())
        classifier_checkpoint = torch.load("./models/checkpoint_msra_AL_2/3_2.pth")
        classifier.load_state_dict(classifier_checkpoint)

    # ----------------------------------------train----------------------------------------
    for epoch in range(args.start_epoch, args.epochs + 1):
        train(train_loader, model, classifier, criterion, optimizer, epoch)
        model_name = args.save_path + str(epoch) + '_1.pth'
        classifier_name = args.save_path + str(epoch) + '_2.pth'
        model.module.save(model_name)
        classifier.save(classifier_name)
        lfw_eval.eval(model_eval, model_name)
        evaluate_prototype.eval(model_eval, epoch,
             model_name,
             args.eval_root,
             args.eval_list,
             device,
             batch_size=400, workers=12)
    print('Finished Training')


def train(train_loader, model, classifier, criterion, optimizer, epoch):
    model.train()
    print_with_time('Epoch {} start training'.format(epoch))
    time_curr = time.time()
    loss_display = 0.0

    for batch_idx, (data, target) in enumerate(train_loader, 1):
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        adjust_learning_rate(optimizer, iteration, args.step_size)
        data, target = data.to(device), target.to(device)
        # compute output
        output = model(data)
        if isinstance(classifier, torch.nn.Linear):
            output = classifier(output)
        else:
            output = classifier(output, target)
        loss = criterion(output, target)
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
