import torch
import torch.nn as nn
from torch.nn import functional as F
import geomloss

class ProjLayer(nn.Module):
    '''
    inputs: features of encoder block
    outputs: projected features
    '''
    def __init__(self, in_c, out_c):
        super(ProjLayer, self).__init__()
        self.proj = nn.Sequential(nn.Conv2d(in_c, in_c//2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//2, in_c//4, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//4),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//4, in_c//2, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(in_c//2),
                                  torch.nn.LeakyReLU(),
                                  nn.Conv2d(in_c//2, out_c, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm2d(out_c),
                                  torch.nn.LeakyReLU(),
                                  )
    def forward(self, x):
        return self.proj(x)
    
class MultiProjectionLayer(nn.Module):
    def __init__(self, base = 64):
        super(MultiProjectionLayer, self).__init__()
        self.proj_a = ProjLayer(base * 4, base * 4)
        self.proj_b = ProjLayer(base * 8, base * 8)
        self.proj_c = ProjLayer(base * 16, base * 16)
    def forward(self, features, features_noise = False):
        if features_noise is not False:
            return ([self.proj_a(features_noise[0]),self.proj_b(features_noise[1]),self.proj_c(features_noise[2])], \
                  [self.proj_a(features[0]),self.proj_b(features[1]),self.proj_c(features[2])])
        else:
            return [self.proj_a(features[0]),self.proj_b(features[1]),self.proj_c(features[2])]

def loss_fucntion(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(a[item].shape[0],-1),
                                      b[item].view(b[item].shape[0],-1)))
    return loss


def loss_concat(a, b):
    mse_loss = torch.nn.MSELoss()
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    a_map = []
    b_map = []
    size = a[0].shape[-1]
    for item in range(len(a)):
        a_map.append(F.interpolate(a[item], size=size, mode='bilinear', align_corners=True))
        b_map.append(F.interpolate(b[item], size=size, mode='bilinear', align_corners=True))
    a_map = torch.cat(a_map,1)
    b_map = torch.cat(b_map,1)
    loss += torch.mean(1-cos_loss(a_map,b_map))
    return loss


class CosineReconstruct(nn.Module):
    def __init__(self):
        super(CosineReconstruct, self).__init__()
    def forward(self, x, y):
        return torch.mean(1 - torch.nn.CosineSimilarity()(x, y))

class Revisit_RDLoss(nn.Module):
    """
    receive multiple inputs feature
    return multi-task loss:  SSOT loss, Reconstruct Loss, Contrast Loss
    """
    def __init__(self, consistent_shuffle = True):
        super(Revisit_RDLoss, self).__init__()
        self.sinkhorn = geomloss.SamplesLoss(loss='sinkhorn', p=2, blur=0.05, \
                              reach=None, diameter=10000000, scaling=0.95, \
                                truncate=10, cost=None, kernel=None, cluster_scale=None, \
                                  debias=True, potentials=False, verbose=False, backend='auto')
        self.reconstruct = CosineReconstruct()       
        self.contrast = torch.nn.CosineEmbeddingLoss(margin = -0.5)
    def forward(self, noised_feature, projected_noised_feature, projected_normal_feature):
        """
        noised_feature : output of encoder at each_blocks : [noised_feature_block1, noised_feature_block2, noised_feature_block3]
        projected_noised_feature: list of the projection layer's output on noised_features, projected_noised_feature = projection(noised_feature)
        projected_normal_feature: list of the projection layer's output on normal_features, projected_normal_feature = projection(normal_feature)
        """
        current_batchsize = projected_normal_feature[0].shape[0]

        target = -torch.ones(current_batchsize).to('cuda')

        normal_proj1 = projected_normal_feature[0]
        normal_proj2 = projected_normal_feature[1]
        normal_proj3 = projected_normal_feature[2]
        # shuffling samples order for caculating pair-wise loss_ssot in batch-mode , (for efficient computation)
        shuffle_index = torch.randperm(current_batchsize)
        # Shuffle the feature order of samples in each block
        shuffle_1 = normal_proj1[shuffle_index]
        shuffle_2 = normal_proj2[shuffle_index]
        shuffle_3 = normal_proj3[shuffle_index]

        abnormal_proj1, abnormal_proj2, abnormal_proj3 = projected_noised_feature
        noised_feature1, noised_feature2, noised_feature3 = noised_feature
        loss_ssot = self.sinkhorn(torch.softmax(normal_proj1.view(normal_proj1.shape[0], -1), -1), torch.softmax(shuffle_1.view(shuffle_1.shape[0], -1),-1)) +\
               self.sinkhorn(torch.softmax(normal_proj2.view(normal_proj2.shape[0], -1),-1),  torch.softmax(shuffle_2.view(shuffle_2.shape[0], -1),-1)) +\
               self.sinkhorn(torch.softmax(normal_proj3.view(normal_proj3.shape[0], -1),-1),  torch.softmax(shuffle_3.view(shuffle_3.shape[0], -1),-1))
        loss_reconstruct = self.reconstruct(abnormal_proj1, normal_proj1)+ \
                   self.reconstruct(abnormal_proj2, normal_proj2)+ \
                   self.reconstruct(abnormal_proj3, normal_proj3)
        loss_contrast = self.contrast(noised_feature1.view(noised_feature1.shape[0], -1), normal_proj1.view(normal_proj1.shape[0], -1), target = target) +\
                           self.contrast(noised_feature2.view(noised_feature2.shape[0], -1), normal_proj2.view(normal_proj2.shape[0], -1), target = target) +\
                           self.contrast(noised_feature3.view(noised_feature3.shape[0], -1), normal_proj3.view(normal_proj3.shape[0], -1), target = target)
        return (loss_ssot + 0.01 * loss_reconstruct + 0.1 * loss_contrast)/1.11
