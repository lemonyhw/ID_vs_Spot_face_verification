from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.autograd import Variable
import math

def cosine_sim(x1, x2, dim=1, eps=1e-8):
    ip = torch.mm(x1, x2.t()) #
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return ip / torch.ger(w1,w2).clamp(min=eps)

class MarginCosineProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.35):
        super(MarginCosineProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        #stdv = 1. / math.sqrt(self.weight.size(1))
        #self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, label):
        cosine = cosine_sim(input, self.weight)
        # cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        # --------------------------- convert label to one-hot ---------------------------
        # https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = self.s * (cosine - one_hot * self.m)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features,m=4):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        # self.weight.data = weight
        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)

class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature
            m: margin
            cos(theta + m)
        """
    def __init__(self, in_features, out_features, weight,s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # nn.init.xavier_uniform_(self.weight)
        self.weight.data = weight
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)
        return output


def get_one_hot(labels, num_classes):
    one_hot = Variable(torch.range(0, num_classes - 1)).unsqueeze(0).expand(labels.size(0), num_classes)
    one_hot = one_hot.cuda()
    one_hot = one_hot.eq(labels.unsqueeze(1).expand_as(one_hot).float()).float()
    return one_hot
#Dummpy loss
class DumLoss(nn.Module):
    def __init__(self,in_features,num_classes,weight):
        super(DumLoss, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_features,num_classes,bias=False)
        self.classifier.weight.data = weight
        self.softmax = nn.Softmax()
    # def forward(self,input,label):
    #     # print("label",label)
    #     one_hot = get_one_hot(label,self.num_classes)
    #     # print("weight",weight)
    #     # print("input",input)
    #     p = self.softmax(self.classifier(input))
    #     penalty_coef = torch.abs((one_hot - p))
    #     # print("penalty_coef",penalty_coef.size(),penalty_coef)
    #     #1. probablities  2. penalty.
    #     tp2 = torch.mm(self.classifier.weight,input.t())
    #     # print("tp2",tp2.size(),tp2)
    #     Ldum = torch.mul(penalty_coef,tp2.t())
    #     # print("Ldum",Ldum.size(),Ldum)
    #     Ldum = torch.abs(Ldum).sum()
    #     # torch.cuda.empty_cache()
    #     # print("loss",loss)
    #     return Ldum

    def forward(self, input,label):
        one_hot = get_one_hot(label, self.num_classes)
        output = self.classifier(input)
        p = self.softmax(output)
        penalty_coef = one_hot - p
        pwx = penalty_coef.mul(output)
        return torch.abs(torch.sum(pwx))


def cross_entropy(logits, target, size_average=True):
    if size_average:
        return torch.mean(torch.sum(-target * F.log_softmax(logits,-1), -1))
    else:
        return torch.sum(torch.sum(-target * F.log_softmax(logits, -1), -1))


class NpairLoss(nn.Module):
    """the multi-class n-pair loss"""
    def __init__(self, l2_reg=0.02):
        super(NpairLoss, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchor, positive, target):
        batch_size = anchor.size(0)
        target = target.view(target.size(0), 1)
        target = (target == torch.transpose(target, 0, 1)).float()
        target = target / torch.sum(target, dim=1, keepdim=True).float()
        logit = torch.matmul(anchor, torch.transpose(positive, 0, 1))

        loss_ce = cross_entropy(logit, target)
        l2_loss = torch.sum(anchor ** 2) / batch_size + torch.sum(positive ** 2) / batch_size
        loss = loss_ce + self.l2_reg * l2_loss * 0.25
        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """
    def __init__(self, margin=2.0):
        super(ContrastiveLoss,self).__init__()
        self.margin = margin
    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label).float() * torch.pow(euclidean_distance,2)+label.float()*torch.pow(torch.clamp(self.margin - euclidean_distance,min=0.0),2))
        return loss_contrastive


class DIAMSoftmaxLoss(nn.Module):
    def __init__(self, in_features, out_features,weight,device):
        super(DIAMSoftmaxLoss, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.weight.data = weight
        # nn.init.xavier_uniform_(self.weight)
        self.m = 0.35
        self.s = 30
        self.alpha = 0.5
        self.device = device
        # for p in self.parameters():
        #     p.requires_grad = False

    def forward(self, input, label):
        # normalize
        weight_norm = F.normalize(self.weight) #[100000,512]
        feature_norm = F.normalize(input)      #[200,512]

        bs = feature_norm.size(0)
        nc = self.out_features
        #
        dist_mat_glob = feature_norm.mm(weight_norm.transpose(-2, 1))
        # dist_mat_glob = dist_mat_glob.byte()

        label = label.view(-1,1)
        label_mat_glob1 = torch.zeros(bs, nc).to(self.device).scatter_(1, label, 1)
        label_mat_glob2 = torch.ones(bs, nc).to(self.device).scatter_(1, label, 0)

        label_mask_pos_glob = label_mat_glob1.byte()
        label_mask_neg_glob = label_mat_glob2.byte()

        logits_pos_glob = torch.masked_select(dist_mat_glob, label_mask_pos_glob)
        logits_neg_glob = torch.masked_select(dist_mat_glob, label_mask_neg_glob)

        _logits_pos = logits_pos_glob.view(bs, -1)
        _logits_neg = logits_neg_glob.view(bs, -1)

        _logits_pos = _logits_pos * self.s
        _logits_neg = _logits_neg * self.s

        _logits_neg = torch.log(torch.sum(torch.exp(_logits_neg), dim=1))
        _logits_neg = _logits_neg.view(bs, -1)

        t = self.m + _logits_neg - _logits_pos
        loss = torch.log(1 + torch.exp(t))
        loss = torch.mean(loss)


        #Dynamic weight imprinting
        #We follow the CenterLoss to update the weights, which is equivalent to
        #imprinting the mean features

        weight_update = self.weight.data
        label_unique = torch.unique(label, sorted=False)

        for idx in reversed(label_unique):
            one_hot = torch.eq(label, idx)
            num = one_hot.sum()
            one_hot = one_hot.expand(bs,self.in_features)
            weight_mask = torch.masked_select(feature_norm, one_hot) #weight_norm[200,512]
            weight_mask = weight_mask.view(num, -1) #[num,512]
            weight_update[idx,:] = weight_mask.mean(0)
        self.weight.data = (1-self.alpha) * self.weight.data + self.alpha * weight_update
        return loss

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)



class DWIAngleLinear(nn.Module):
    def __init__(self, in_features, out_features, weight, m=4):
        super(DWIAngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.alpha = 1
        self.weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        # nn.init.xavier_uniform_(self.weight)
        self.weight.data = weight
        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(theta) & phi(theta) ---------------------------
        bs = input.size(0)
        feature_norm = F.normalize(input)
        cos_theta = F.linear(feature_norm, F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        weight_update = self.weight.data
        label_unique = torch.unique(label, sorted=False)
        label_ = label.view(-1,1)
        for idx in reversed(label_unique):
            one_hot = torch.eq(label_, idx)
            num = one_hot.sum()
            one_hot = one_hot.expand(bs, self.in_features)
            weight_mask = torch.masked_select(feature_norm, one_hot)  #
            weight_mask = weight_mask.view(num, -1)  # [num,512]
            weight_update[idx, :] = weight_mask.mean(0)
        self.weight.data = (1 - self.alpha) * self.weight.data + self.alpha * weight_update
        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            torch.save(self.state_dict(), f)