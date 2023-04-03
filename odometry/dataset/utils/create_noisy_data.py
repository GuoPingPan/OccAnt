import numpy as np
import os.path as osp
from glob import glob
import cv2
import os
from tqdm import tqdm

import sys
sys.path.append("/home/yzc1/workspace/OccAnt/odometry")

# from habitat_env.add_obs_noise import RGBNoise
from dataset.habitat_env.add_obs_noise import RGBNoise, DepthNoise

class NoiseObsGenerator:
    def __init__(self, data_dir, split):
        dataset_dir = osp.join(data_dir, split)
        scene_names = glob(dataset_dir + '/*')
        self.rgb_imgs = []
        self.depth_imgs = []

        self.add_rgb_noise = RGBNoise(intensity_constant=0.1)
        self.add_depth_noise = DepthNoise(noise_multiplier=0.5)


        for scene in scene_names:
            self.rgb_imgs = self.load_img(scene, '.jpg')
            self.depth_imgs = self.load_img(scene, '.npy')

            rgb_out_path = osp.join(scene, 'rgb_noise')
            depth_out_path = osp.join(scene, 'depth_noise')

            os.makedirs(rgb_out_path, exist_ok=True)
            assert osp.exists(rgb_out_path), f"make file {rgb_out_path} failed."
            os.makedirs(depth_out_path, exist_ok=True)
            assert osp.exists(depth_out_path), f"make file {depth_out_path} failed."

            for rgb_img_pth, depth_img_pth in tqdm(zip(self.rgb_imgs, self.depth_imgs)):
                rgb_img = cv2.imread(rgb_img_pth, -1)
                depth_img = np.load(depth_img_pth)

                rgb_img_name = rgb_img_pth.split('/')[-1]
                depth_img_name = depth_img_pth.split('/')[-1]
                assert rgb_img_name.split('.')[0] == depth_img_name.split('.')[0]
                
                noisy_rgb_img = self.add_rgb_noise(rgb_img)
                noisy_depth_img = self.add_rgb_noise(depth_img)

                cv2.imwrite(osp.join(rgb_out_path, rgb_img_name), noisy_rgb_img)
                np.save(osp.join(depth_out_path, depth_img_name), noisy_depth_img)


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

    
if __name__ == "__main__":
    NoiseObsGenerator("/home/yzc1/workspace/OccAnt/data/vo_dataset", split="train")
    NoiseObsGenerator("/home/yzc1/workspace/OccAnt/data/vo_dataset", split="val")
    NoiseObsGenerator("/home/yzc1/workspace/OccAnt/data/vo_dataset", split="val_mini")
