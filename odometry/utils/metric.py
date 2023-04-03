import numpy as np
import torch
import torch.nn.functional as F


def rot_and_trans_mae(pred, pose_gt):
    pose_est = pred['p_delta'].clone().detach()
    
    trans_mae = torch.mean((pose_est[..., :-1] - pose_gt[..., :-1]).abs()).cpu() 
    rot_mae = torch.mean((pose_est[..., -1] - pose_gt[..., -1]).abs()).cpu() 

    return trans_mae, rot_mae

def rot_and_trans_nig_uncer(pred):
    uncer = (pred['beta'] / (pred['v']*(pred['alpha'] - 1))).clone().detach()
    trans_uncer = uncer[..., :-1].cpu()
    rot_uncer = uncer[..., -1].cpu()

    return torch.mean(trans_uncer, dim=0), torch.mean(rot_uncer, dim=0)

def rot_and_trans_laplacian_uncer(pred):
    sigma = pred['p_sigma'].clone().detach()
    trans_uncer = sigma[..., :-1].cpu()
    rot_uncer = sigma[..., -1].cpu()

    return torch.mean(trans_uncer, dim=-1), torch.mean(rot_uncer, dim=-1)

def mae_mean_max_torch(pred, gt):
    mask = (gt > 0.1) & (gt < 10.)
    mae = torch.sum((pred - gt).abs()*mask, dim=[-2, -1]) / torch.sum(mask, dim=[-2, -1])
    return mae.mean(), mae.max()

def mae_mean_max_numpy(pred, gt):
    mask = (gt > 0.1) & (gt < 10.)
    mae = np.sum(np.sum(np.abs(pred - gt)*mask, axis=-1), axis=-1) / np.sum(np.sum(mask, axis=-1), axis=-1)
    return np.mean(mae), np.max(mae)

def sigma_torch(pred, gt, t=1):
    mask = (gt > 0.1) & (gt < 10.)
    percent = ((torch.max(pred / gt, gt / pred)< 1.25**t)*mask).sum()  \
                / torch.sum(mask)

    return percent

def sigma_numpy(pred, gt, t=1):
    mask = (gt > 0.1) & (gt < 10.)
    percent = np.sum((np.maximum(pred / gt, gt / pred) < 1.25 ** t) * mask) \
              / np.sum(mask)

    return percent

# a = np.random.rand(10,10,20,20)
# b = np.random.rand(10,10,20,20)
# print(np.sum(a, axis=-1).shape)
# print(np.sum(np.sum(a, axis=-1), axis=-1).shape)
# print(mae_mean_max_numpy(a, b))
# print(sigma_numpy(a, b))
#
#
# a = torch.rand(10,10,20,20)
# b = torch.rand(10,10,20,20)
# print(mae_mean_max_torch(a, b))
# print(sigma_torch(a, b))

# sum 有 dim，两个tensor大小没有dim
# print(torch.max(a, b, dim=(-2, -1)).shape)
# print(per)