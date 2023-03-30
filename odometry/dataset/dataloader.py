import json
import numpy as np
import os.path as osp
from glob import glob
import gzip
from PIL import Image
import cv2

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

# from habitat_env.add_obs_noise import RGBNoise
from .habitat_env.add_obs_noise import RGBNoise

class VoDataset(Dataset):
    def __init__(self, data_dir, split, add_obs_noise=False):
        dataset_dir = osp.join(data_dir, split)
        scene_names = glob(dataset_dir + '/*')
        self.rgb_imgs = []
        self.depth_imgs = []
        self.action_ids = []
        self.collisions = []
        self.pose_delta = []
        self.transform = transforms.ToTensor()
        self.add_obs_noise = add_obs_noise

        for scene in scene_names:
            self.rgb_imgs += self.load_img(scene, '.jpg')
            self.depth_imgs += self.load_img(scene, '.npy')
            action_ids, pose_delta, collisions = self.from_json(scene)
            self.action_ids += action_ids
            self.pose_delta += pose_delta
            self.collisions += collisions

        if add_obs_noise:
            self.add_rgb_noise = RGBNoise(intensity_constant=0.1)


    def load_img(self, scene, prefix):
        '''
        Return:
            img_path：List[str]
        '''
        if prefix == '.jpg':
            img_path = glob(osp.join(scene, 'rgb/*.jpg'))
        elif prefix == '.npy':
            img_path = glob(osp.join(scene, 'depth/*.npy'))
        else:
            raise FileExistsError(f"prefix don't exist: {prefix}")

        img_path = sorted(img_path)
        return img_path

    def from_json(self, scene):
        '''
        Return:
            action_ids：List[int]
            pose_delta：List[List[float]]
        '''
        action_ids, pose_delta, collisions = [], [], []
        with gzip.open(osp.join(scene, 'pose.json.gz'), 'rt') as f:
            data = json.load(f)
            for item in data:
                action_ids.append(item['action_id'])
                pose_delta.append(item['pose_delta'])
                collisions.append(item['collision'])
        return action_ids, pose_delta, collisions

    def __len__(self):
        return len(self.action_ids)

    def __getitem__(self, idx):
        data = {}
        rgb_t_1 = cv2.imread(self.rgb_imgs[2 * idx], -1)
        rgb_t = cv2.imread(self.rgb_imgs[2 * idx + 1], -1)
        rgb_t_1 = cv2.cvtColor(rgb_t_1, cv2.COLOR_BGR2RGB)
        rgb_t = cv2.cvtColor(rgb_t, cv2.COLOR_BGR2RGB)
        if self.add_obs_noise:
            rgb_t_1 = self.add_rgb_noise(rgb_t_1)
            rgb_t = self.add_rgb_noise(rgb_t)
        # print(type(rgb_t_1), rgb_t_1.shape)
        rgb_t_1 = self.transform(Image.fromarray(rgb_t_1))
        rgb_t = self.transform(Image.fromarray(rgb_t))
        depth_t_1 = torch.from_numpy(np.load(self.depth_imgs[2 * idx])).permute(2, 0, 1) * 9.9 + 0.1
        depth_t = torch.from_numpy(np.load(self.depth_imgs[2 * idx + 1])).permute(2, 0, 1) * 9.9 + 0.1
        collision = torch.tensor(self.collisions[idx]).long()
        action = torch.tensor(self.action_ids[idx]).long()
        pose_delta = torch.tensor(self.pose_delta[idx])
        data['rgb_t_1'] = rgb_t_1
        data['rgb_t'] = rgb_t
        data['depth_t_1'] = depth_t_1
        data['depth_t'] = depth_t
        data['pose_delta'] = pose_delta
        data['action_id'] = action
        data['collision'] = collision

        return data

    @staticmethod
    def collate_fn(batch):
        if isinstance(batch[0], dict):
            batch_size = len(batch)
            data = {}
            keys = ['rgb_t_1', 'rgb_t', 'depth_t_1', 'depth_t', 'pose_delta', 'action_id', 'collision']
            for key in keys:
                data[key] = torch.stack([batch[i][key] for i in range(batch_size)], dim=0)

            return data

        else:
            return batch



# if __name__ =='__main__':
#     a = Image.fromarray(np.random.rand(244,244))
#     print(transforms.ToTensor()(a).shape)

#     dataset = VoDataset(data_dir='/home/yzc1/workspace/OccAnt/data/vo_dataset', split='train', add_obs_noise=True)
#     from torch.utils.data import DataLoader
#     dataloader = DataLoader(dataset,
#                              batch_size=64,
#                              shuffle=True,
#                              num_workers=8,
#                              pin_memory=True,
#                             collate_fn=VoDataset.collate_fn)
#     for i in dataloader:
#         # print(i)
#         exit()