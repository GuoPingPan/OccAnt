import torch
import torch.nn as nn
import utils.transform as transform
import numpy as np

class PoseL1Loss(nn.Module):
    def __init__(self, rot_wight=5):
        super().__init__()

        self.l1 = nn.SmoothL1Loss(reduction='mean')
        self.rot_wight = rot_wight

    def forward(self, pred, pose_gt,):

        pose_est = pred['p_delta']
        pose_l1_lose = self.l1(pose_est[..., :-1], pose_gt[..., :-1]) \
            + self.rot_wight * self.l1(pose_est[..., -1], pose_gt[..., -1])

        return pose_l1_lose

class PoseGaussianLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, pose_gt):

        pose_est = pred['p_delta']
        sigma = pred['p_sigma']

        e = (pose_est - pose_gt)
        e_T = e.unsqueeze(1)
        exp_term = (e_T.matmul(sigma.inverse()).matmul(e.unsqueeze(-1))).mean()
        det = (torch.linalg.det(sigma)+1).log().mean() # may < 0
        pose_uncer_loss = exp_term + det

        return pose_uncer_loss



class PoseNIGLossOld(nn.Module):
    def __init__(self, split = True, beta: float = 5.):
        super().__init__()
        self.l1 = nn.SmoothL1Loss(reduction='mean')
        self.beta = beta
        self.split = split

    def forward(self, pose_nig, pose_gt, ):

        mu, alpha, v, beta = pose_nig['p_delta'], pose_nig['alpha'], pose_nig['v'], pose_nig['beta']

        Omega = 2*beta*(1 + v)

        nll_loss = 0.5*torch.log(torch.pi/v) - alpha*torch.log(Omega) \
                 + (alpha + 0.5)*torch.log(v*(mu-pose_gt)**2 + Omega) + \
                    torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

        error = (mu - pose_gt).abs()
        evi = 2*v + alpha
        reg = error*evi


        if self.split:
            loss = (nll_loss[..., :-1] + reg[..., :-1]).mean() + self.beta * (nll_loss[..., -1] + reg[..., -1]).mean()

        else:
            loss = (nll_loss + reg).mean()

        return loss


class PoseLaplacianUncerLoss(nn.Module):
    def __init__(self, beta: float=.05):
        super().__init__()
        self.beta = beta

    def forward(self, pred, pose_gt):

        pose_est = pred['p_delta']
        sigma = pred['p_sigma']

        pose_uncer_loss = (sigma - 1 + torch.exp(-(pose_est-pose_gt).abs()
                                                  # / (self.beta * (pose_est+pose_gt))
                                                 / self.beta
                                                 )
                           ).abs().mean()

        return pose_uncer_loss


class PoseUncerAuxilaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pose_est, pose_ref, sigma, type='inverse'):

        pose_uncer_loss = - 2*(pose_est.unsqueeze(1) \
                            .matmul(sigma.inverse()) \
                            .matmul(pose_ref.unsqueeze(-1))).mean()
        # if type == 'inverse':
        #     pass
        # elif type == 'flip':
        #     pose_uncer_loss *= -1.
        # else:
        #     raise TypeError

        return pose_uncer_loss # may < 0


class PoseNIGLoss(nn.Module):

    def __init__(self, lam=0.0, epsilon=1e-2, maxi_rate=1e-4):
        super().__init__()

        self.lam = lam
        self.epsilon = epsilon
        self.maxi_rate = maxi_rate

    def _nig_nll(self, y, gamma, v, alpha, beta, reduce=True):
        Omega = 2*beta*(1+v)

        nll = 0.5*torch.log(np.pi/v)  \
            - alpha*torch.log(Omega)  \
            + (alpha+0.5) * torch.log(v*(y-gamma)**2 + Omega)  \
            + torch.lgamma(alpha)  \
            - torch.lgamma(alpha+0.5)

        return torch.mean(nll) if reduce else nll

    def _kl_nig(self, mu1, v1, a1, b1, mu2, v2, a2, b2):
        KL = 0.5*(a1-1)/b1 * (v2*torch.square(mu2-mu1))  \
            + 0.5*v2/v1  \
            - 0.5*torch.log(torch.abs(v2)/torch.abs(v1))  \
            - 0.5 + a2*torch.log(b1/b2)  \
            - (torch.lgamma(a1) - torch.lgamma(a2))  \
            + (a1 - a2)*torch.digamma(a1)  \
            - (b1 - b2)*a1/b1
        return KL

    def _nig_reg(self, y, gamma, v, alpha, beta, omega=0.01, reduce=True, kl=False):
        # error = torch.stop_gradient(torch.abs(y-gamma))
        error = torch.abs(y-gamma)

        if kl:
            kl = self._kl_nig(gamma, v, alpha, beta, gamma, omega, 1+omega, beta)
            reg = error*kl
        else:
            evi = 2*v+(alpha)
            reg = error*evi

        return torch.mean(reg) if reduce else reg

    def forward(self, pose_nig, pose_gt, reduce=True):

        mu, alpha, v, beta = pose_nig['p_delta'], pose_nig['alpha'], pose_nig['v'], pose_nig['beta']
      
        nll_loss = self._nig_nll(pose_gt, mu, v, alpha, beta, reduce=reduce)
        reg_loss = self._nig_reg(pose_gt, mu, v, alpha, beta, reduce=reduce)
        loss = nll_loss + self.lam * (reg_loss - self.epsilon)
        
        return loss, reg_loss


class DepthLoss(nn.Module):
    def __init__(self, depth_min, depth_max):
        super().__init__()

        self.l1loss = nn.SmoothL1Loss(reduction='sum')
        self.depth_min = depth_min
        self.depth_max = depth_max

    def _get_smooth_loss(self, disp, img):
        """Computes the smoothness loss for a disparity image
        The color image is used for edge-aware smoothness

        Key:
            注意要使用 Depth 对应的那张 rgb
        """
        mean_disp = torch.mean(disp, dim=(-2,-1), keepdim=True)
        disp_temp = disp / (mean_disp + 1e-7)

        grad_disp_x = torch.abs(disp_temp[:, :, :, :-1] - disp_temp[:, :, :, 1:]).to(disp.device)
        grad_disp_y = torch.abs(disp_temp[:, :, :-1, :] - disp_temp[:, :, 1:, :]).to(disp.device)

        grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True).to(disp.device)
        grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True).to(disp.device)

        grad_disp_x *= torch.exp(-grad_img_x)
        grad_disp_y *= torch.exp(-grad_img_y)

        return grad_disp_x.mean() + grad_disp_y.mean()

    def forward(self, disp_est, depth_gt, rgb):
        depth_est = transform.disp_to_uniform_depth(disp_est, depth_min=self.depth_min, depth_max=self.depth_max)
        mask = (depth_gt > 0.1) & (depth_gt < 10.)

        # print("depth_est_mean: ", depth_est.mean())
        # print("depth_est shape: ", depth_est.shape)
        # print("mask_mean: ", mask.float().mean())
        # print("depth_gt_mean: ", depth_gt.mean())
        # print("mask sum", torch.sum(mask, dim=[2, 3]))
        # print("mask sum", torch.sum((depth_est - depth_gt).abs()*mask, dim=[2, 3]))

        # depth_l1_loss = self.l1loss(depth_est*mask, depth_gt*mask) / torch.sum(mask, dim=[2, 3])
        depth_l1_loss = (torch.sum((depth_est - depth_gt).abs()*mask, dim=[2, 3]) / (torch.sum(mask, dim=[2, 3]) + 1e-4)).mean()
        # depth_smooth_loss = self._get_smooth_loss(disp_est, rgb)
        # print('depth_l1_loss: ', depth_l1_loss)
        # print(f'depth_loss: {depth_l1_loss}        depth_smooth_loss: {depth_smooth_loss}')
        # assert (depth_l1_loss + depth_smooth_loss) > 0, f'depth_smooth_loss: {depth_smooth_loss}, depth_l1_loss:{depth_l1_loss}'
        # return depth_l1_loss + depth_smooth_loss
        return depth_l1_loss

class DepthUncerLoss(nn.Module):
    def __init__(self, depth_min, depth_max, beta=.2):
        super().__init__()

        self.depth_min = depth_min
        self.depth_max = depth_max
        self.beta = beta

    def forward(self, d_1, d_2, depth_uncer, type='origin'):
        depth = transform.disp_to_uniform_depth(d_1, depth_min=self.depth_min, depth_max=self.depth_max)

        if type == 'origin':
            # depth_uncer_loss = (depth - d_2).abs()*(1 - depth_uncer)
            depth_uncer_loss = (depth_uncer - 1 + torch.exp(-(depth - d_2).abs()
                                                            # / (self.beta*(depth + d_2))
                                                            # / self.beta
                                                            )
                                ).abs() + (depth - d_2).abs()*(1 - depth_uncer)


        elif type == 'flip':
            depth_flip = transform.disp_to_uniform_depth(d_2, depth_min=self.depth_min, depth_max=self.depth_max)

            depth_uncer_loss = (depth_uncer - 1 + torch.exp(-(depth - depth_flip).abs()
                                                            # / (self.beta*(depth + depth_flip))
                                                            # / self.beta
                                                            )
                                ).abs() + (depth - depth_flip).abs()*(1 - depth_uncer)

        else:
            raise TypeError
        # print(f'depth_uncer_loss: {depth_uncer_loss.mean()}')

        return depth_uncer_loss.mean()

