import collections
import numpy as np
from abc import ABC
import torch
import torch.nn.functional as F
from torch import nn, autograd
from .losses import CrossEntropyLabelSmooth
from IPython import embed


class CM(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update
        for x, y in zip(inputs, targets):
            ctx.features[y] = ctx.momentum * ctx.features[y] + (1. - ctx.momentum) * x
            ctx.features[y] /= ctx.features[y].norm()

        return grad_inputs, None, None, None


def cm(inputs, indexes, features, momentum=0.5):
    return CM.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))


class CM_Hybrid(autograd.Function):

    @staticmethod
    def forward(ctx, inputs, targets, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(inputs, targets)
        outputs = inputs.mm(ctx.features.t())

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets = ctx.saved_tensors
        nums = len(ctx.features)//2
        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        batch_centers = collections.defaultdict(list)
        for instance_feature, index in zip(inputs, targets.tolist()):
            batch_centers[index].append(instance_feature)

        for index, features in batch_centers.items():
            distances = []
            for feature in features:
                distance = feature.unsqueeze(0).mm(ctx.features[index].unsqueeze(0).t())[0][0]
                distances.append(distance.cpu().numpy())

            mean = torch.stack(features, dim=0).mean(0)
            ctx.features[index] = ctx.features[index] * ctx.momentum + (1 - ctx.momentum) * mean
            ctx.features[index] /= ctx.features[index].norm()
            
            hard = np.argmin(np.array(distances))
            ctx.features[index+nums] = ctx.features[index+nums] * ctx.momentum + (1 - ctx.momentum) * features[hard]
            ctx.features[index+nums] /= ctx.features[index+nums].norm()


        return grad_inputs, None, None, None


def cm_hybrid(inputs, indexes, features, momentum=0.5):
    return CM_Hybrid.apply(inputs, indexes, features, torch.Tensor([momentum]).to(inputs.device))



class ClusterMemory(nn.Module, ABC):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2, mode='CM', smooth=0, has_multi_pos_loss=False):
        super(ClusterMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp
        self.cm_type = mode
        self.has_multi_pos_loss = has_multi_pos_loss

        if smooth > 0:
            self.cross_entropy = CrossEntropyLabelSmooth(self.num_samples, 0.1, True)
            print('>>> Using CrossEntropy with Label Smoothing.')
        else:
            self.cross_entropy = nn.CrossEntropyLoss().cuda()

        if self.cm_type == 'CM':
            self.register_buffer('features', torch.zeros(num_samples, num_features))
        elif self.cm_type == 'CMhybrid':
            self.register_buffer('features', torch.zeros(2 * num_samples, num_features))           
        else:
            raise TypeError('Cluster Memory {} is invalid!'.format(self.cm_type))


    def forward(self, inputs, targets):

        inputs = F.normalize(inputs, dim=1).cuda()
        if self.cm_type == 'CM':
            outputs = cm(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            loss = self.cross_entropy(outputs, targets)
            return loss

        elif self.cm_type == 'CMhybrid':
            outputs = cm_hybrid(inputs, targets, self.features, self.momentum)
            outputs /= self.temp
            mean, hard = torch.chunk(outputs, 2, dim=1)

            if self.has_multi_pos_loss:
                loss = 0.5 * (self.get_multi_pos_loss(hard, targets) + self.get_multi_pos_loss(mean, targets))
            else:
                r = 0.2
                loss = 0.5 * (self.cross_entropy(hard, targets) + torch.relu(self.cross_entropy(mean, targets) - r))
            return loss


    def get_multi_pos_loss(self, score, batch_proxy_ind):
        assert(self.proxy_pseudo_labels is not None)
        temp_score = score.detach().clone()
        bg_knn = 50
        loss = 0
        for i in range(len(score)):
            pseudo_lbl = self.proxy_pseudo_labels[batch_proxy_ind[i]]
            pos_ind = torch.nonzero(self.proxy_pseudo_labels == pseudo_lbl).squeeze(-1)
            assert(len(pos_ind)>=1 and len(pos_ind)<3)
            temp_score[i, pos_ind] = 10000
            _, sel_ind = torch.topk(temp_score[i], k=bg_knn)
            sel_score = score[i, sel_ind]
            sel_target = torch.zeros(sel_score.shape, dtype=sel_score.dtype).cuda()
            sel_target[0:len(pos_ind)] = 1.0/len(pos_ind)
            loss += -1.0 * (F.log_softmax(sel_score.unsqueeze(0), dim=1) * sel_target.unsqueeze(0)).sum()
        loss /= len(score)

        return loss
