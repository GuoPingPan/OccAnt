import torch
from torchvision import transforms
import shutil
import numpy as np
import gym
import os
import os.path as osp
import cv2
import random
import json
import gzip
import math
from tqdm import tqdm

import habitat
import quaternion
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from .add_obs_noise import RGBNoise

absolute_path_prefix = os.getcwd()+'../../'


def rotation_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return x, y, z

def get_relative_pose(pre, cur):

    t1 = pre.position
    t2 = cur.position

    r1 = quaternion.as_rotation_matrix(pre.rotation)
    r2 = quaternion.as_rotation_matrix(cur.rotation)

    rr = np.linalg.inv(r1).dot(r2)
    tt = np.linalg.inv(r1).dot((t2-t1).reshape(-1,1)).reshape(-1)

    do = rotation_to_euler_angles(rr)[1]

    return [-tt[2], -tt[0], tt[1], do]


class DatasetEnv(habitat.RLEnv):
    def __init__(self, args, rank, config_env, dataset=None):

        self.args = args
        self.rank = rank
        self.pairs_per_scene = args.pairs_per_scene
        self.episodes_per_scene = args.episodes_per_scene
        self.min_steps_per_episode = int(np.ceil(self.pairs_per_scene * 1.0 / self.episodes_per_scene) + 2)

        self.action_space = gym.spaces.Discrete(3)
        self.action_map = {1: 'forward', 2: 'left', 3: 'right'}

        # key: used in VectorEnv
        self.original_action_space = self.action_space
        self.observation_space = gym.spaces.Box(0, 255, (3, args.frame_height, args.frame_width), dtype='uint8')
        
        self.number_of_scenes = len(config_env.DATASET.CONTENT_SCENES)

        super(DatasetEnv, self).__init__(config_env, dataset)

        goal_radius = self.episodes[0].goals[0].radius
        if goal_radius is None:
            goal_radius = config_env.SIMULATOR.FORWARD_STEP_SIZE

        self.shortest_path_follower = ShortestPathFollower(self.habitat_env.sim, goal_radius, False)

        self.generate_dataset_output_dir = self.args.generate_dataset_output_dir.format(split=self.args.split)
        self.trajectory_out_dir = self.args.trajectory_out_dir
        # assert not os.makedirs(self.generate_dataset_output_dir, exist_ok=True)
        # assert not os.makedirs(self.trajectory_out_dir, exist_ok=True)
        print(f'Saving datasets to {self.generate_dataset_output_dir}')
        print(f'Saving trajectories to {self.trajectory_out_dir}')

    def reset(self):

        obs = super().reset()
        self.last_habitat_location = self.habitat_env.sim.get_agent_state(0)

        return obs

    def step(self, action):

        # Action remapping
        # if action == 1:  # Forward
        #     noisy_action = 'noisy_forward'
        # elif action == 3:  # Right
        #     noisy_action = 'noisy_right'
        # elif action == 2:  # Left
        #     noisy_action = 'noisy_left'
        # else:  # stop
        #     noisy_action = 0
        # # print(f'action: {self.action_map[action]}')

        # if not self.args.no_noisy_actions:
        #     # print('noisy')
        #     obs, rew, done, info = super().step(noisy_action)
        # else:
        obs, rew, done, info = super().step(action)

        return obs, rew, done, info

    # def get_habitat_sim_loc_and_turn_to_self_coord(self):
    def get_rel_pose_change(self):
        '''This function is get the agent location in habitat sim
        and turn to self coordinate
        Concept:
            habitat_sim coord: [x,y,z] = agent[right, up, back]
                   self coord: [x,y,z] = habitat[-z,-x,y] = agent[front, left, up]
        '''

        agent_state = self.habitat_env.sim.get_agent_state(0)
        delta_pose = get_relative_pose(self.last_habitat_location, agent_state)
        self.last_habitat_location = agent_state

        return delta_pose

    def run(self):

        for i in range(self.number_of_scenes):
            self.rgb_same_scene, self.depth_same_scene, self.pose_same_scene, self.action_same_scene, self.collision_same_scene = [], [], [], [], []

            num_episode_count = 0

            while num_episode_count < self.episodes_per_scene:
                rgb_seq, depth_seq, pose_seq, action_seq, collision_seq = [], [], [], [], []


                obs = self.reset()
                rgb_seq.append(obs['rgb'])
                depth_seq.append(obs['depth'])

                goal_position = self.habitat_env.current_episode.goals[0].position
                start_position = self.habitat_env.current_episode.start_position


                while not self.habitat_env.episode_over:

                    best_action = self.shortest_path_follower.get_next_action(goal_position)
                    if best_action is None or best_action == 0:
                        break

                    obs, rew, done, info = self.step(best_action)

                    # agent coordinate pose
                    pose = self.get_rel_pose_change()
                    assert pose[-1] > -1. and pose[-1] < 1., "error"

                    ''' check collision effect '''
                    # print('\n')
                    # print("action: ", self.action_map[best_action])
                    # print("pose: ", pose)
                    # print("info: ", info)
                    # print('\n')

                    rgb_seq.append(obs['rgb'])
                    depth_seq.append(obs['depth'])
                    action_seq.append(best_action)
                    pose_seq.append(pose)
                    collision_seq.append(info['collisions']['is_collision'])


                # step threshold for episode
                if self.habitat_env._elapsed_steps < self.min_steps_per_episode:
                    continue

                if num_episode_count == 0:
                    self.scene_ids = self.habitat_env.current_episode.scene_id.split('/')[-1].split('.')[0]
                    print(f"********************** Scene id: {self.scene_ids} **********************")
                    print(f'\t Each scene use episodes: {self.args.episodes_per_scene} , pairs: {self.args.pairs_per_scene}')

                num_episode_count += 1

                self.rgb_same_scene.append(rgb_seq)
                self.depth_same_scene.append(depth_seq)
                self.pose_same_scene.append(pose_seq)
                self.action_same_scene.append(action_seq)
                self.collision_same_scene.append(collision_seq)

            print(f'\t Total num of episodes: {len(self.rgb_same_scene)}')
            assert len(self.rgb_same_scene) > 0, "all episode useless"

            self.save_dataset()

    def save_dataset(self):

        assert len(self.rgb_same_scene) == len(
            self.depth_same_scene), "rgb scene length doesn't match depth scene length"

        scene_dir = osp.join(self.generate_dataset_output_dir, self.scene_ids)
        if os.path.exists(scene_dir):
            shutil.rmtree(scene_dir)

        assert not os.makedirs(scene_dir, exist_ok=True), f'make dir [{scene_dir}] failed!'

        rgb_dir = osp.join(scene_dir, 'rgb')
        depth_dir = osp.join(scene_dir, 'depth')
        assert not os.makedirs(rgb_dir, exist_ok=True), f'make dir [{rgb_dir}] failed!'
        assert not os.makedirs(depth_dir, exist_ok=True), f'make dir [{depth_dir}] failed!'

        num_of_episodes = len(self.rgb_same_scene)
        pairs_per_epoch = int(np.ceil(self.pairs_per_scene / num_of_episodes))
        assert pairs_per_epoch < self.min_steps_per_episode, f"pairs per epoch: {pairs_per_epoch} > minimum: {self.min_steps_per_episode}"

        count = 0
        json_dataset = []

        for j in range(num_of_episodes):
            index = list(range(len(self.rgb_same_scene[j]) - 1))
            random.shuffle(index)
            # print(f'len(index): {len(index)}    pairs_per_epoch: {pairs_per_epoch}    self.min_steps_per_episode: {self.min_steps_per_episode}')

            assert len(self.rgb_same_scene[j]) == len(self.rgb_same_scene[j]), "rgb doesn't match depth"
            assert len(index) > pairs_per_epoch, "out of index"

            for id in index[:pairs_per_epoch]:
                # key: cv2读入的时候会默认将rgb转成bgr，写出的时候也是将bgr转变成rgb写出，因此需要转成bgr才能够正确调用cv2的接口完成写出
                img_t = cv2.cvtColor(self.rgb_same_scene[j][id], cv2.COLOR_RGB2BGR)
                img_t_1 = cv2.cvtColor(self.rgb_same_scene[j][id + 1], cv2.COLOR_RGB2BGR)
                cv2.imwrite(osp.join(rgb_dir, f"{count:04d}.jpg"), img_t)
                cv2.imwrite(osp.join(rgb_dir, f"{count + 1:04d}.jpg"), img_t_1)
                np.save(osp.join(depth_dir, f"{count:04d}"), self.depth_same_scene[j][id])
                np.save(osp.join(depth_dir, f"{count + 1:04d}"), self.depth_same_scene[j][id + 1])

                data = {}
                data['action_id'] = int(self.action_same_scene[j][id])
                data['pose_delta'] = [float(element) for element in self.pose_same_scene[j][id]]
                data['action'] = self.action_map[int(self.action_same_scene[j][id])]
                data['collision'] = bool(self.collision_same_scene[j][id])
                json_dataset.append(data)

                count += 2

        with open(osp.join(scene_dir, 'pose.json'), 'wt') as f:
            json.dump(json_dataset, f)

        with gzip.open(osp.join(scene_dir, 'pose.json.gz'), 'wt') as f:
            json.dump(json_dataset, f)

        print(f"Finished save dataset about scenes: {self.scene_ids}")



    def turn_pose_to_grip_map(self, pose, map_size = 5):
        '''
        :param map_size: 5cm
        '''

        grid_pose = (pose - self.init_pose)
        translation = (grid_pose[:3])*100 / map_size

        return translation[1], translation[0]

    def draw_trajectory(self, pose, color):
        y, x = self.turn_pose_to_grip_map(pose)
        #         y_center, x_center = self.grip_map.shape[0] / 2, self.grip_map.shape[1] / 2
        #
        #         y = int(y + y_center)
        #         x = int(x + x_center)
        #         # print(y, x)
        #
        #         # self.grip_map[y, x] = color
        cv2.circle(self.grip_map, (x, y), 3, color)

    def get_ros_coord_pose(self):
        agent_state = self.habitat_env.sim.get_agent_state(0)
        position = agent_state.position

        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        # 对于左边 axis = 0,
        if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        # 对于右边 axis = 3.14 or -3.14
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return np.array([-position[2], -position[0], position[1], o])

    def get_pose_from_delta(self, delta):
        self.angle_w2a = (self.last_pose)[-1]

        self.Rw2a = np.array([
            [np.cos(self.angle_w2a), -np.sin(self.angle_w2a)],
            [np.sin(self.angle_w2a), np.cos(self.angle_w2a)]
        ])

        self.last_pose[:2] += self.Rw2a.dot(delta[:2])
        self.last_pose[2:] += delta[2:]


        if self.last_pose[-1] > np.pi:
            self.last_pose[-1] -= 2 * np.pi

        elif self.last_pose[-1] < - np.pi:
            self.last_pose[-1] += 2 * np.pi

    def test(self, model, device='cuda'):

        num_episode_count = 0
        self.add_rgb_noise = RGBNoise(intensity_constant=0.1)
        self.transform = transforms.ToTensor()
        all_trajectory_error = 0

        ids = np.random.choice(self.episodes_per_scene, size=50).tolist()

        while num_episode_count < self.episodes_per_scene:

            obs = self.reset()

            rgb_seq = [obs['rgb']]
            depth_seq = [obs['depth']]
            self.init_pose = self.get_ros_coord_pose()
            self.last_pose = np.copy(self.init_pose)

            # 300 * 5 / 100 = 15 m
            self.grip_map = np.zeros(shape=(400, 400, 3), dtype=np.uint8)
            # bgr
            # gt 是红色, pred 是蓝色
            gt_color = (0, 0, 255)
            pred_color = (255, 0, 0)

            self.draw_trajectory(self.init_pose, gt_color)

            goal_position = self.habitat_env.current_episode.goals[0].position
            start_position = self.habitat_env.current_episode.start_position

            count = 0
            trajectory_error = 0


            while not self.habitat_env.episode_over:

                best_action = self.shortest_path_follower.get_next_action(goal_position)

                if best_action is None or best_action == 0:
                    break

                obs, rew, done, info = self.step(best_action)
                rgb_seq.append(obs['rgb'])
                depth_seq.append(obs['depth'])
                collision = info['collisions']['is_collision']

                gt_pose = self.get_ros_coord_pose()
                self.draw_trajectory(gt_pose, gt_color)


                # use vonet
                if self.add_rgb_noise:
                    rgb_seq[-2] = self.add_rgb_noise(rgb_seq[-2])
                    rgb_seq[-1] = self.add_rgb_noise(rgb_seq[-1])
                rgb_t_1 = self.transform(rgb_seq[-2])
                rgb_t = self.transform(rgb_seq[-1])
                rgb_input = torch.concat([rgb_t_1, rgb_t], dim=0).unsqueeze(0).to(device)

                best_action_tensor = torch.tensor(best_action).long().unsqueeze(0).to(device)
                collision_tensor = torch.tensor(collision).long().unsqueeze(0).to(device)

                pose_nig_est, disp_est, depth_uncer_est = model(rgb_input, best_action_tensor, collision_tensor)

                delta = pose_nig_est['mu'].cpu().detach().numpy().squeeze()
                self.get_pose_from_delta(delta)

                trajectory_error += np.abs(self.last_pose - gt_pose)

                self.draw_trajectory(self.last_pose, pred_color)

                count += 1

            num_episode_count += 1
            trajectory_error_mean = trajectory_error / count
            print(f"[{num_episode_count} / {self.episodes_per_scene}]")
            # print(f"steps: {count}  trajectory_error: {trajectory_error_mean}")
            # print(f"                trajectory_length: {self.last_pose - self.init_pose}")
            all_trajectory_error += trajectory_error_mean

            if num_episode_count in ids:
                cv2.imwrite(f"{osp.join(self.trajectory_out_dir, f'{num_episode_count}')}.png", self.grip_map)
            # else:
            #     exit()
        print(f"num_episodes: {num_episode_count}   all_trajectory_error: {all_trajectory_error / num_episode_count}")


    # habitat-lab/habitat-lab/habitat/core/environments.py
    def get_reward(self, observations):
        return 0.

    def get_done(self, observations):
        done = False
        if self.habitat_env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    def get_reward_range(self):
        # This function is not used, Habitat-RLEnv requires this function
        return (0., 1.0)

    def get_spaces(self):
        return self.observation_space, self.action_space

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space