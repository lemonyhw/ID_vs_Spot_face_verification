import numpy as np
import torch
import json
import torchvision.transforms as transforms
import itertools
from utils.dataset_evaluate import PImageList
from utils import net_msra
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

def evaluate_xjc(scores, labels):
    # scores = scores.detach().cpu().numpy()
    # labels = actual_issame.detach().cpu().numpy()
    index = np.lexsort((labels, scores))
    scores = scores[index]
    labels = labels[index]
    pos = np.sum(labels)
    neg = labels.shape[0] - pos

    TPR = np.cumsum(labels[::-1])[::-1]
    FPR = np.cumsum((1-labels)[::-1])[::-1]
    acc = TPR+(neg-FPR)
    acc = acc/(neg+pos)

    bestAcc = np.max(acc)
    # bestAcc = 0.5
    bestThresh = np.where(acc==bestAcc)[0][-1]

    # print("TPR",TPR.size)
    # TPR_atBestThresh = TPR[bestThresh]/pos
    # FPR_atBestThresh = FPR[bestThresh]/neg
    # APCER = FPR[bestThresh]/neg
    # NPCER = (pos-TPR[bestThresh])/pos
    # ACER = (APCER+NPCER)/2
    # bestThresh = scores[bestThresh]

    # print(bestThresh)
    pre_TPR = TPR/pos
    pre_FPR = FPR/neg

    # TPR@FPR=10e-2
    FPR_001 = np.where(pre_FPR>=0.01)[0][-1]
    TPR_001 = pre_TPR[FPR_001]
    # Thresh_at01 = scores[FPR_001]

    # TPR@FPR=10e-3
    FPR_0001 = np.where(pre_FPR >= 0.001)[0][-1]
    TPR_0001 = pre_TPR[FPR_0001]
    # Thresh_at001 = scores[FPR_001]

    # TPR@FPR=10e-4
    FPR_00001 = np.where(pre_FPR >= 0.0001)[0][-1]
    TPR_00001 = pre_TPR[FPR_00001]
    # Thresh_at0001 = scores[FPR_0001]

    # TPR@FPR=10e-5
    FPR_000001 = np.where(pre_FPR >= 0.00001)[0][-1]
    TPR_000001 = pre_TPR[FPR_000001]
    # Thresh_at00001 = scores[FPR_00001]

    # return bestAcc, bestThresh, APCER, NPCER, ACER, TPR_01,Thresh_at01,TPR_001,Thresh_at001,TPR_0001,Thresh_at0001,TPR_00001,Thresh_at00001
    return TPR_001,TPR_0001,TPR_00001,TPR_000001


def extractFeature(model,eval_root, pair, batch_size, workers, device):
    transform = transforms.Compose([
        transforms.CenterCrop(128),
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    loader = torch.utils.data.DataLoader(
        PImageList(root=eval_root, train_root=pair,
                   transform=transform),
        batch_size=int(batch_size / 2), shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False)
    Spot_feat = []
    ID_feat = []
    with torch.no_grad():
        for batch_idx, (Spot, ID, Spot_fp, ID_fp) in enumerate(loader, 1):
            Spot, ID, Spot_fp, ID_fp = Spot.to(device), ID.to(device), Spot_fp.to(device), ID_fp.to(device)
            Spot_op, ID_op, Spot_fp_op, ID_fp_op = model(Spot), model(ID), model(Spot_fp), model(ID_fp)
            Spot_ft = torch.cat((Spot_op, Spot_fp_op), 1)
            ID_ft = torch.cat((ID_op, ID_fp_op), 1)
            if batch_idx == 1:
                Spot_feat = Spot_ft
                ID_feat = ID_ft
            else:
                Spot_feat = torch.cat((Spot_feat, Spot_ft), 0)
                ID_feat = torch.cat((ID_feat, ID_ft), 0)
    return Spot_feat, ID_feat

def compute_distacne(Spot_ft, ID_ft, pair_set, batch_size, sameflag=1):
    predicts = []
    length = pair_set.shape[0]
    batch_num = int(length / batch_size)
    pair_Spot = pair_set[:, 0]
    pair_ID = pair_set[:, 1]
    for i in range(batch_num):
        if (i + 1) * batch_size <= length:
            pair_Spot_b = pair_Spot[i * batch_size:(i + 1) * batch_size]
            pair_ID_b = pair_ID[i * batch_size:(i + 1) * batch_size]
            Spot_ft_b = Spot_ft[pair_Spot_b]
            ID_ft_b = ID_ft[pair_ID_b]
        else:
            pair_Spot_b = pair_Spot[i * batch_num:]
            pair_ID_b = pair_ID[i * batch_num:]
            Spot_ft_b = Spot_ft[pair_Spot_b]
            ID_ft_b = ID_ft[pair_ID_b]
        dot = torch.sum(Spot_ft_b.mul(ID_ft_b), dim=1)
        distance = dot / (Spot_ft_b.norm(dim=1) * ID_ft_b.norm(dim=1) + 1e-5)
        distance = distance.cpu().numpy()
        predicts.extend(distance)
    if sameflag == 1:
        label = np.ones(len(predicts))
    else:
        label = np.zeros(len(predicts))
    label = label.reshape(-1, 1)
    predicts = np.array(predicts)
    predicts = predicts.reshape(-1, 1)
    return [predicts, label]

def eval(model,
         epoch,
         model_path,
         eval_root,
         eval_list,
         device,
         batch_size=400, workers=12):
    model.module.load_state_dict(torch.load(model_path))
    model.eval()

    with open(eval_list) as json_file:
        data = json.load(json_file)
    pair_s = data["image_names"]
    array = np.arange(len(pair_s))
    set_all = list(itertools.product(array, repeat=2))
    set_d = list(itertools.permutations(array, 2))
    set_s = list(set(set_all) - set(set_d))
    set_d = np.array(set_d)
    set_s = np.array(set_s)

    # print("Extract Feature..")
    Spot_feat, ID_feat = extractFeature(model,eval_root, pair_s, batch_size, workers, device)

    predicts_T = compute_distacne(Spot_feat, ID_feat, set_s, batch_size, sameflag=1)
    predicts_F = compute_distacne(Spot_feat, ID_feat, set_d, batch_size, sameflag=0)

    predict = np.vstack((predicts_T[0],predicts_F[0]))

    label = np.vstack((predicts_T[1],predicts_F[1]))
    predict = np.reshape(predict,-1,1)
    label = np.reshape(label,-1,1)

    TPR_001, TPR_0001, TPR_00001, TPR_000001 = evaluate_xjc(predict, label)
    line_vote_acc = "{}\t{:.6}\t{:.6}\t{:.6f}\t{:.6}\n".format(epoch,
                                                               TPR_001,
                                                               TPR_0001,
                                                               TPR_00001,
                                                               TPR_000001)

    f2 = open("TPR_NJN_DUM.txt", "a+")
    f2.write(line_vote_acc)



if __name__ == '__main__':
    device = torch.device("cuda")
    model = net_msra.sphere(type=64)
    model = torch.nn.DataParallel(model).cuda()
    model_path = '/home/yanghuiwen/project/3_low_shot_learning/Large-scale_Bisample_Learning_on_ID_vs_Spot_Face_Recognition/models/checkpoint_NJN_RP/DummpyFace_5_checkpoint.pth'
    eval_root = '/home3/yhw_datasets/face_recognition/NJN_crop/test'
    eval_list = '/home/yanghuiwen/project/3_low_shot_learning/Large-scale_Bisample_Learning_on_ID_vs_Spot_Face_Recognition/Data/NJN_Random_prototype_test.json'
    epoch = 4000
    result = eval(model,epoch,model_path,eval_root,eval_list,device,batch_size=400, workers=12)

    # np.savetxt("result.txt",result,'%s')


