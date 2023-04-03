import sys
import time

import torch
import shutil
import os
import os.path as osp
os.environ['HABITAT_SIM_LOG'] = "quiet"
os.environ['MAGNUM_LOG'] = "quiet"
sys.path.append('/home/hello/pgp_eai/easan')

from habitat.config import get_config

from models.vo import VoNet
from dataset.habitat_env.env import DatasetEnv
from arguments import get_args

absolute_path_prefix = os.getcwd() + '/../'


def init_model(args):
    model = VoNet(num_layers=args.num_layers,
                        frame_width=args.frame_width,
                        frame_height=args.frame_height,
                        decoder_type=args.decoder_type,
                        split_action=args.split_action,
                        after_compression_flat_size=args.after_compression_flat_size,
                        p_dropout=args.p_dropout,
                        use_dropout=args.use_dropout,
                        num_input_images=args.num_input_images,
                        pose_type=args.pose_type,
                        action_space=args.action_space,
                        emb_layers=args.emb_layers,
                        hidden_size=args.hidden_size,
                        use_act_embedding=args.use_act_embedding,
                        use_collision_embedding=args.use_collision_embedding,
                        embedding_size=args.embedding_size,
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
    config = get_config(args.config)
    scenes = ['Greigsville']

    config.defrost()
    config.DATASET.SPLIT = args.split
    config.DATASET.CONTENT_SCENES = scenes
    config.DATASET.MAX_SCENE_REPEAT_STEPS = args.max_episode_length
            
    # add absoulte path
    config.DATASET.DATA_PATH = absolute_path_prefix + config.DATASET.DATA_PATH
    config.DATASET.SCENES_DIR = absolute_path_prefix + config.DATASET.SCENES_DIR
    config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = args.episodes_per_scene

    config.freeze()

    env = DatasetEnv(args, 0, config_env=config)

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
