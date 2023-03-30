import sys
import time

import torch
import shutil
import os
import os.path as osp
os.environ['HABITAT_SIM_LOG'] = "quiet"
os.environ['MAGNUM_LOG'] = "quiet"
sys.path.append('/home/hello/pgp_eai/easan')

from habitat.config import read_write, get_config
from habitat.config.default_structured_configs import CollisionsMeasurementConfig

from OccAnt.odometry.models.vo import VoNet
from dataset.dataset_env import Dataset_Env
from arguments import get_args

absolute_path_prefix = os.getcwd()


def init_model(args):
    model = VoNet(num_layers=args.num_layers,
                        frame_width=args.frame_width,
                        frame_height=args.frame_height,
                        after_compression_flat_size=args.after_compression_flat_size,
                        hidden_size=args.hidden_size,
                        p_dropout=args.p_dropout,
                        num_input_images=args.num_input_images,
                        pose_type=args.pose_type,
                        pretrained=args.en_pre,
                        use_group_norm=args.use_group_norm)


    if args.vonet_checkpoint is not None:
        model.load_state_dict(torch.load(args.vonet_checkpoint))
    else:
        raise FileNotFoundError('Please load the pretrained model for valid!')

    model.to(args.device)
    model.eval()

    return model


def init_env(args):
    config = get_config(config_path=args.config)
    scenes = ['Greigsville']

    with read_write(config):
        config.habitat.dataset.split = args.split
        config.habitat.dataset.content_scenes = scenes
        config.habitat.environment.max_episode_steps = args.max_episode_length

        # add absoulte path
        config.habitat.dataset.data_path = osp.join(absolute_path_prefix, config.habitat.dataset.data_path)
        config.habitat.dataset.scenes_dir = osp.join(absolute_path_prefix, config.habitat.dataset.scenes_dir)
        config.habitat.environment.iterator_options.max_scene_repeat_episodes = args.episodes_per_scene

        if not args.no_collision:
            config.habitat.task.measurements.collision = CollisionsMeasurementConfig()

    env = Dataset_Env(args, 0, config_env=config)

    return env

if __name__ == '__main__':
    args = get_args()

    output_dir = args.trajectory_out_dir

    if osp.exists(output_dir):
        sig = input(f'output_dir is not empty: {output_dir} \n'
                    f'if delete it ?[y|n]\t')
        if sig == 'y' or sig == 'Y':
            shutil.rmtree(output_dir)
            print("Successfully delete!")
        else:
            exit('Please change the output_dir!')

    assert not os.makedirs(output_dir, exist_ok=True), f'make dir [{output_dir}] failed!'

    model = init_model(args)
    env = init_env(args)
    start = time.time()
    env.test(model, args.device)
    end = time.time()
    print(f"Visualization use time: {start - end}")
