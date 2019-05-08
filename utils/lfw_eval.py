from PIL import Image
import numpy as np

from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import torch
import torch.backends.cudnn as cudnn

cudnn.benchmark = True
from utils import net
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
def extractDeepFeature(img, model):
    transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    img1 = transform(img)
    img2 = transform(F.hflip(img))
    img1 = img1.unsqueeze(0).to('cuda')
    img2 =img2.unsqueeze(0).to('cuda')
    ft = torch.cat((model(img1), model(img2)), 1)[0].to('cpu')
    return ft

def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_true = []
    y_predict = []
    for d in diff:
        same = 1 if float(d[2]) > threshold else 0
        y_predict.append(same)
        y_true.append(int(d[3]))
    y_true = np.array(y_true)
    y_predict = np.array(y_predict)
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

def eval(model, model_path=None):
    predicts = []
    model.module.load_state_dict(torch.load(model_path))
    model.eval()
    root = '/home3/yhw_datasets/face_recognition/lfw_crop_1/'
    with open('/home3/yhw_datasets/face_recognition/pairs.txt') as f:
        pairs_lines = f.readlines()[1:]

    with torch.no_grad():
        for i in range(6000):
            p = pairs_lines[i].replace('\n', '').split('\t')
            if 3 == len(p):
                sameflag = 1
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
            elif 4 == len(p):
                sameflag = 0
                name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
                name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
            else:
                raise ValueError("WRONG LINE IN 'pairs.txt! ")

            with open(root + name1, 'rb') as f:
                img1 =  Image.open(f).convert('RGB')
            with open(root + name2, 'rb') as f:
                img2 =  Image.open(f).convert('RGB')
            f1 = extractDeepFeature(img1, model)
            f2 = extractDeepFeature(img2, model)

            distance = f1.dot(f2) / (f1.norm() * f2.norm() + 1e-5)
            predicts.append('{}\t{}\t{}\t{}\n'.format(name1, name2, distance, sameflag))

    accuracy = []
    thd = []
    folds = KFold(n=6000, n_folds=10)
    thresholds = np.arange(-1.0, 1.0, 0.005)
    predicts = np.array(list(map(lambda line: line.strip('\n').split(), predicts)))
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
        thd.append(best_thresh)
    print('LFWACC={:.4f} std={:.4f} thd={:.4f}'.format(np.mean(accuracy), np.std(accuracy), np.mean(thd)))
    return np.mean(accuracy), predicts


if __name__ == '__main__':
    model = net.sphere(type=64)
    model = torch.nn.DataParallel(model).to(device)
    model_path = './checkpoint-webface_asoftmax/SphereFace64_1_checkpoint.pth'
    # checkpoint = torch.load(model_path)
    # model.module.load_state_dict(checkpoint)
    _, result = eval(model, model_path=model_path)
    np.savetxt("result.txt", result, '%s')
