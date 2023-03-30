import os
import os.path as osp

import torch

import numpy as np
import argparse

absolute_path_prefix = os.getcwd()


def get_args():
    os.environ['MAGNUM_LOG'] = "quiet"
    os.environ['GLOG_minloglevel'] = "2"
    os.environ['KMP_WARNINGS'] = "off"

    parser = argparse.ArgumentParser(description='LASAN')


    ''' Common Parameters '''
    parser.add_argument('-d', '--dataset', required=True, type=str,
                        help='the path of datasets')

    parser.add_argument('--device', default='cpu',
                        help='device id (i.e. 0 or 0,1 or cuda)')

    parser.add_argument("--split", type=str, default="train",
                        help="dataset split (train | val | val_mini) ")



    ''' Env Parameters '''
    parser.add_argument("--config", type=str,
                        default=osp.join(absolute_path_prefix,
                                         "config/gibson_obs_noise_free_and_actuator_noise.w_collision.gt_loc.yaml"),
                        help="path to config yaml containing task information")

    parser.add_argument('--add_obs_noise', action='store_true', default=False)
    parser.add_argument('--total_num_scenes', type=str, default="auto")


    ''' GPU setting '''
    parser.add_argument('-agc', '--auto_gpu_config', type=int, default=1)
    parser.add_argument('--num_processes', type=int, default=0)

    parser.add_argument('-n', '--num_processes_per_gpu', type=int, default=0,
                        help="""how many processes per gpu to use
                                Overridden when auto_gpu_config=1and training on gpus """)

    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')



    ''' Train Parameters '''
    parser.add_argument('--num_epsisodes', type=int, default=1000,
                        help='training episodes')
    parser.add_argument('-el', '--max_episode_length', type=int, default=1000,
                        help="maximum episode length for each episode")
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--policy_log_interval', type=float, default=100)
    parser.add_argument('--save_policy_per_episodes', type=float, default=200)

    # training active env
    parser.add_argument('--map_size', type=int, default=480,
                        help='(default=480pixels equal to 24m with resolution=5)')
    parser.add_argument('--padding', type=int, default=0)
    parser.add_argument('--map_resolution', type=int, default=5,
                        help="grid map resolution (default=5cm)")
    parser.add_argument('--no_uncer', action='store_true', default=False)
    parser.add_argument('--uncer_thres', type=float, default=0.5)
    parser.add_argument('--occ_threshold', type=float, default=5.,
                        help="the occupancy threshold (default = 5 point)")
    parser.add_argument('--vision_range', type=float, default=5.,
                        help="the range of camera is (default = 5 m)")




    ''' Generate Dataset Parameters '''
    parser.add_argument('-gdo', '--generate_dataset_output_dir', type=str,
                        default='/home/yzc1/workspace/OccAnt/data/vo_dataset/{split}')
    parser.add_argument('-pps', '--pairs_per_scene', type=int, default=200,
                        help="collect pairs of data each scene")
    parser.add_argument('-epp', '--episodes_per_scene', type=int, default=100,
                        help="num of episodes for each scene")


    ''' Test Map Parameters'''
    parser.add_argument('-mvo','--map_visualize_out_dir', type=str,
                        help='The path for pred depth img result of the VoNet')


    ''' Test Depth Prediction Parameters'''
    parser.add_argument('-tdo','--test_depth_out_dir', type=str,
                        help='The path for pred depth img result of the VoNet')
    parser.add_argument('-ntp', '--num_test_pic', type=int, default=30)


    ''' Test Trajectory Prediction Parameters'''
    parser.add_argument('-tro' ,'--trajectory_out_dir', type=str,
                        default='/home/yzc1/workspace/OccAnt/output/trajectory/temp',
                        help='the path of trajectory')


    ''' VoNet Parameters '''
    # train
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=2.5e-4)
    parser.add_argument('--lr_step', type=float, default=1e3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--vonet_log_interval', type=float, default=10)
    parser.add_argument('-vckpt', '--vonet_checkpoint', type=str, default=None, help='load checkpoint')
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--save_model_epochs', type=int, default=5)


    # encoder
    parser.add_argument('--num_layers', type=int, default=18)
    parser.add_argument('-ugn', '--use_group_norm', action='store_true', default=False)
    parser.add_argument('--en_pre', action='store_true', default=False,
                        help='resnet encoder use pretrained model')
    parser.add_argument('--num_input_images', type=int, default=2)
    parser.add_argument('-fw', '--frame_width', type=int, default=128,
                        help='network frame width (default:84)')
    parser.add_argument('-fh', '--frame_height', type=int, default=128,
                        help='network frame height (default:84)')

    # pose decoder
    parser.add_argument('--pose_type', type=str, default='SE2')
    parser.add_argument('--hidden_size', type=list, default=[256,256])
    parser.add_argument('--p_dropout', type=float, default=0.2)
    parser.add_argument('--use_dropout', action='store_true', default=False)
    parser.add_argument('--after_compression_flat_size', type=int, default=1024)
    parser.add_argument('--embedding_size', type=int, default=8)
    parser.add_argument('--split_action', action='store_true', default=False)
    parser.add_argument('--decoder_type', type=str, default="base")
    parser.add_argument('--action_space', type=int, default=3)
    parser.add_argument('--emb_layers', type=int, default=2)
    parser.add_argument('--use_act_embedding', action='store_true', default=False)
    parser.add_argument('--use_collision_embedding', action='store_true', default=False)


    ''' Active Policy Parameters '''
    parser.add_argument('-pckpt', '--policy_checkpoint', type=str, default=None, help='load checkpoint')
    parser.add_argument('--not_train_global', action='store_true', default=False,
                        help='load checkpoint')
    parser.add_argument('--global_obs_size', type=int, default=240,
                        help='global_observation_size')
    parser.add_argument('--global_downscale', type=int, default=2,
                        help='global_downscale')
    parser.add_argument('--global_hidden_size', type=int, default=256,
                        help='global_hidden_size')
    parser.add_argument('--use_recurrent_global', type=int, default=1,
                        help='use a recurrent global policy')

    parser.add_argument('--global_lr', type=float, default=2.5e-5,
                        help='global learning rate (default: 2.5e-5)')
    parser.add_argument('--eps', type=float, default=1e-5,
                        help='RL Optimizer epsilon (default: 1e-5)')
    parser.add_argument('--alpha', type=float, default=0.99,
                        help='RL Optimizer alpha (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--use_gae', action='store_true', default=False,
                        help='use generalized advantage estimation')
    parser.add_argument('--tau', type=float, default=0.95,
                        help='gae parameter (default: 0.95)')
    parser.add_argument('--entropy_coef', type=float, default=0.001,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--value_loss_coef', type=float, default=0.5,
                        help='value loss coefficient (default: 0.5)')
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument('--num_global_steps', type=int, default=1000,
                        help='number of forward steps in A2C (default: 5)')
    parser.add_argument('--ppo_epoch', type=int, default=4,
                        help='number of ppo epochs (default: 4)')
    parser.add_argument('--ppo_mini_batch_size', type=int, default=1,
                        help='number of batches for ppo (default: 32)')
    parser.add_argument('--clip_param', type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')


    args = parser.parse_args()
    allocate_gpu(args)

    return args


def allocate_gpu(args):

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")

    if args.cuda:

        if args.total_num_scenes != "auto":
            args.total_num_scenes = int(args.total_num_scenes)
        elif "gibson" in args.config and "train" in args.split:
            args.total_num_scenes = 72
        elif "gibson" in args.config and "val_mini" in args.split:
            args.total_num_scenes = 3
        elif "gibson" in args.config and "val" in args.split:
            args.total_num_scenes = 14
        else:
            assert False, "Unknown task config, please specify\n total_num_scenes"

        args.num_gpus = 1

        if not args.auto_gpu_config:
            # 手动设置的时候均等分
            if not args.num_processes_per_gpu:
                raise ValueError(f'please step up the next params manually, or turn auto_gpu_config\n'
                                    f'1.num_processes_per_gpu\n')

            args.num_processes = args.num_gpus*args.num_processes_per_gpu

            args.scenes_per_process = max(1, int(args.total_num_scenes // args.num_processes))

            # 这里的想法是对于每个gpu分别有多少个线程，而对于手动分配的时候每个gpu线程数是相同的
            args.process_split = [args.num_processes_per_gpu for i in range(args.num_gpus)]
            args.total_num_scenes = args.scenes_per_process*args.num_processes

            print("Manual GPU config:")

        else:
            '''
                Automatically configure number of training threads based on
                number of GPUs available and GPU memory size
            '''
            args.process_split = []

            for i in range(args.num_gpus):
                # 每个线程好像占用显存就是 1.4GB
                process_per_gpu = int(np.ceil((torch.cuda.get_device_properties(i).total_memory
                                               /1024/1024/1024) / 1.4))
                args.process_split.append(process_per_gpu)

            args.num_processes = np.sum(args.process_split)
            args.scenes_per_process = max(1, int(args.total_num_scenes // args.num_processes))
            args.total_num_scenes = args.scenes_per_process*args.num_processes

            print("Auto GPU config:")

        print("\t Total of processes: {}".format(args.num_processes))
        print("\t Total of scenes: {}".format(args.total_num_scenes))
        print("\t Number of scene per process: {}".format(args.scenes_per_process))
        print("\t Number of processes every GPU: {}".format(args.process_split))

    # run in cpu only use one process
    else:
        args.total_num_scenes = 1
        args.num_processes = 1
        args.num_processes_per_gpu = 1

        print("Using CPU:")
        print("\t Total of processes: {}".format(args.num_processes))
        print("\t Total of scenes: {}".format(args.total_num_scenes))
        print("\t Number of processes per CPU: {}".format(args.num_processes_per_gpu))

