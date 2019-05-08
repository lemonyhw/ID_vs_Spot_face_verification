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
from utils.dataset import PImageList
import json
import random
from utils import net_msra, layer
from utils.evaluate_prototype import eval
import scipy.io as sio

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
torch.backends.cudnn.benchmark = True
# Training settings
parser = argparse.ArgumentParser(description='PyTorch CosFace')

# DATA
parser.add_argument('--root_path', type=str, default='/home3/yhw_datasets/face_recognition/NJN_crop/train',
                    help='path to root path of images')
parser.add_argument('--train_list', type=str, default='./Data/NJN_Random_prototype_train_300000.json',
                    help='path to root path of images')
parser.add_argument('--eval_root', type=str, default="/home3/yhw_datasets/face_recognition/NJN_crop/test")
parser.add_argument('--eval_list', type=str, default='./Data/NJN_Random_prototype_test.json')
parser.add_argument('--database', type=str, default='NJN', help='Which Database for train. (NJN)')
parser.add_argument('--batch_size', type=int, default=200, help='input batch size for training (default: 512)')

# Network
parser.add_argument('--network', type=str, default='sphere64',
                    help='Which network for train. (sphere20, sphere64, LResNet50E_IR)')
parser.add_argument('--pre_model', type=str, default='./models/checkpoint_NJN_Npairloss/50.pth')
parser.add_argument('--new_weight', type=str, default='./Data/Prototype_weight_ID_300000_npair.mat')
# Classifier
parser.add_argument('--num_class', type=int, default=100000, help='number of people(class)')
parser.add_argument('--classifier_type', type=str, default='DWI_AL',
                    help='Which classifier for train. (MCP, AL, L,ARC,TRIP,DUM)')

# Extract feature
parser.add_argument('--prototype', type=str, default="ID", help='which feature for initializing weights')

# LR policy
parser.add_argument('--epochs', type=int, default=30, help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.1)')
parser.add_argument('--step_size', type=list, default=None,
                    help='lr decay step')  # [15000, 22000, 26000][80000,120000,140000][100000, 140000, 160000]
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.9)')
parser.add_argument('--weight_decay', type=float, default=5e-4, metavar='W', help='weight decay (default: 0.0005)')

# Common settings
parser.add_argument('--log_interval', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_path', type=str, default='./models/checkpoint_NJN_RP_DWI_AM_softmax/',
                    help='path to save checkpoint')
parser.add_argument('--no_cuda', type=bool, default=False, help='disables CUDA training')
parser.add_argument('--workers', type=int, default=12, help='how many workers to load data')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.database is "NJN":
    args.step_size = [15000, 30000, 45000]
else:
    raise ValueError("NOT SUPPORT DATABASE! ")


def main():
    # --------------------------------------model----------------------------------------
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
    # pretrain_model = torch.nn.DataParallel(pretrain_model).to(device)
    model = torch.nn.DataParallel(model).to(device)
    model_eval = torch.nn.DataParallel(model_eval).to(device)
    # print(model)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    model.module.save(args.save_path + 'Sphere64_0_checkpoint.pth')

    # ------------------------------------load image---------------------------------------
    # train set
    with open(args.train_list) as json_file:
        data = json.load(json_file)
    train_list = data["image_names"]
    train_label = data["image_labels"]

    train_num = len(train_list)
    print('length of train Database: ' + str(len(train_list)))
    print('Number of Identities: ' + str(args.num_class))
    # --------------------------------Updated prototype matrix-----------------------------
    # Initialize the feature layer
    checkpoint = torch.load(args.pre_model)
    model.module.load_state_dict(checkpoint)

    data = sio.loadmat(args.new_weight)
    new_weight = data["weight"]
    new_weight_ = torch.from_numpy(new_weight)

    list_label = list(zip(train_list, train_label))
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(128),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    # --------------------------------Train-----------------------------
    for epoch in range(1, args.epochs + 1):
        print_with_time('Epoch {} start training'.format(epoch))
        random.shuffle(list_label)
        train_list[:], train_label[:] = zip(*list_label)

        # the whole datasets are divided into len(train_list)/Niter) parts to update
        for subpart in range(int(train_num / args.num_class)):
            print("subpart:", subpart)
            sub_train_list = train_list[subpart * args.num_class:(subpart + 1) * args.num_class]
            sub_train_label = train_label[subpart * args.num_class:(subpart + 1) * args.num_class]
            sub_weight = new_weight_[sub_train_label, :]

            classifier = {
                'MCP': layer.MarginCosineProduct(512, args.num_class).to(device),
                'AL': layer.AngleLinear(512, args.num_class, sub_weight).to(device),
                'L': torch.nn.Linear(512, args.num_class, bias=False).to(device),
                "ARC": layer.ArcMarginProduct(512, args.num_class, sub_weight, s=30, m=0.3, easy_margin=True).to(
                    device),
                'DUM': layer.DumLoss(512, args.num_class, sub_weight).to(device),
                "DWI_AM": layer.DIAMSoftmaxLoss(512, args.num_class, sub_weight, device).to(device),
                "DWI_AL": layer.DWIAngleLinear(512, args.num_class, sub_weight).to(device),
            }[args.classifier_type]
            train_loader = torch.utils.data.DataLoader(
                PImageList(root=args.root_path, train_root=sub_train_list, train_label=sub_train_label,
                           transform=train_transform),
                batch_size=int(args.batch_size / 2), shuffle=True,
                num_workers=args.workers, pin_memory=False, drop_last=True)
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay)
            train(train_loader, model, classifier, optimizer, epoch)

            new_weight_[sub_train_label, :] = classifier.weight.data.cpu()
            temp = classifier.weight.data.cpu()
            # new_weight[sub_train_label,:] =
            for idx, name in enumerate(sub_train_label):
                new_weight[name, :] = temp[idx].numpy()
            # new_weight_ = torch.from_numpy(new_weight)
            model_path = args.save_path + 'DummpyFace_' + str(epoch) + '_checkpoint.pth'
            mat_path = args.save_path + 'DummpyFace_' + str(epoch) + '_checkpoint.mat'

            sio.savemat(mat_path, {"weight": new_weight, "label": train_label})
            model.module.save(model_path)
            eval(model_eval, epoch, model_path, args.eval_root, args.eval_list, device, batch_size=500, workers=12)
    print('Finished Training')


def train(train_loader, model, classifier, optimizer, epoch):
    time_curr = time.time()
    model.train()
    loss_display = 0.0
    for batch_idx, (data1, data2, _, target) in enumerate(train_loader):
        iteration = (epoch - 1) * len(train_loader) + batch_idx
        adjust_learning_rate(optimizer, iteration, args.step_size)
        data1, data2, target = data1.to(device), data2.to(device), target.to(device)

        data = torch.cat([data1, data2], 0)
        target = torch.cat([target, target], 0)

        feats = model(data)
        loss = classifier(feats, target)

        loss_display += loss.item()
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            time_used = time.time() - time_curr
            loss_display /= args.log_interval
            print_with_time(
                'Train Epoch: {} [{}/{}]{}, Loss: {:.6f}, Elapsed time: {:.4f}s({} iters)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset) * 2,
                    iteration, loss_display, time_used, args.log_interval))
            time_curr = time.time()
            loss_display = 0.0
        # torch.cuda.empty_cache()


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
