import torch

def disp_to_depth(disp, depth_min=0.1, depth_max=10.0):
    disp_min = 1.0 / depth_max
    disp_max = 1.0 / depth_min

    depth = 1.0 / (disp_min + disp*(disp_max - disp_min))
    return depth


def image_flip(img: torch.Tensor):
    return torch.flip(img, dims=[-1]).contiguous()


# def disp_to_uniform_depth(disp, depth_min=0.1, depth_max=10.0):
#     depth_uniform = depth_min*(1-disp) \
#                         / (disp*(depth_max - depth_min) + depth_min)
#     return depth_uniform

def disp_to_uniform_depth(disp, depth_min=0.1, depth_max=10.0):
    ran = depth_min * depth_max
    depth_uniform = ran / (disp*(depth_max - depth_min) + depth_min)
    return depth_uniform


class PoseFilp:

    @staticmethod
    def apply(pose_delta: torch.Tensor):
        pose_delta_flip = pose_delta*torch.tensor([1,-1,1,-1]).float().to(pose_delta.device)
        return pose_delta_flip.contiguous()


class PoseInvserse:
    def _theta_to_mat(theta: torch.Tensor):
        mat = torch.stack(
            (
                torch.cos(theta),
                torch.sin(theta),
                -torch.sin(theta),
                torch.cos(theta),
            ),
            dim=1                           
        ).reshape(-1, 2, 2)
        return mat

    @classmethod
    def apply(cls, pose_delta: torch.Tensor):
        theta = pose_delta[:,-1] # n
        mat = cls._theta_to_mat(theta) # n,2,2
        xy = pose_delta[:, :2].unsqueeze(-1) # n,2
        xy_inv = - mat.matmul(xy).reshape(-1,2) # n,2
        # print(xy_inv.shape, (-pose_delta[:, 2].unsqueeze(-1)).shape, (-theta.unsqueeze(-1)).shape)
        pose_delta_invsere = torch.concat(
            (
                xy_inv,
                -pose_delta[:, 2].unsqueeze(-1),
                -theta.unsqueeze(-1)
            ),
            dim=-1
        ).reshape(-1,4)
        return pose_delta_invsere.contiguous()

# a = torch.tensor([[-1,0,-2.5,torch.pi/4]])
# b = PoseInvserse.apply(a)
# c = PoseInvserse.apply(b)
# print(a)
# print(b)
# print(c)


