import random

from habitat.config.default import get_config

from habitat_baselines.common.env_utils import make_env_fn, VectorEnv
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
from habitat import make_dataset


from .habitat_env.env import DatasetEnv

import os

absolute_path_prefix = os.getcwd()+'/../'


def make_env_fn(args, config_env, rank):

    print(f'Process {rank}:')
    print(f'\t Loading scene: {config_env.DATASET.CONTENT_SCENES}')
    dataset = PointNavDatasetV1(config=config_env.DATASET)
    env = DatasetEnv(args=args, rank=rank, config_env=config_env, dataset=dataset)
    env.seed(rank)

    return env

def construct_envs(args, workers_ignore_signals: bool = False,) -> VectorEnv:
    r"""Create VectorEnv object with specified config and env class type.
    To allow better performance, dataset are split into small ones for
    each individual env, grouped by scenes.
    """

    # get all scenes' name
    init_config = get_config(args.config)

    init_config.defrost()
    init_config.DATASET.SPLIT = args.split
    init_config.DATASET.DATA_PATH = absolute_path_prefix + init_config.DATASET.DATA_PATH
    init_config.DATASET.SCENES_DIR = absolute_path_prefix + init_config.DATASET.SCENES_DIR
    # print(config.DATASET)
    init_config.freeze()

    # 这里只是创建了一个空的 dataset 类，
    dataset = make_dataset(init_config.DATASET.TYPE)
    # dataset = PointNavDatasetV1(config_env.dataset)

    # 默认 yaml 不分类，加载所有的 scenes
    scenes = init_config.DATASET.CONTENT_SCENES
    if "*" in init_config.DATASET.CONTENT_SCENES:
        scenes = dataset.get_scenes_to_load(init_config.DATASET)
    scenes = list(scenes)

    finised_scenes = os.listdir(args.generate_dataset_output_dir.format(split=args.split))
    unfinished_scenes = [scene for scene in scenes if scene not in finised_scenes]

    random.shuffle(scenes)
    print(f"All {args.total_num_scenes} scenes:\n", scenes)
    print(f"Finished {len(finised_scenes)} scenes:\n", finised_scenes)
    print(f"Waiting for generate {len(unfinished_scenes)} scenes:\n", unfinished_scenes)

    configs = []
    args_list = []
    scenes_per_process = int(len(unfinished_scenes) / args.num_processes)
    if scenes_per_process == 0:
       scenes_per_process = 1
       args.num_processes_per_gpu = len(unfinished_scenes)
        
    args.process_split = [args.num_processes_per_gpu for i in range(args.num_gpus)]
    


    # # todo: delete
    # scenes = ['Shelbiana', 'Haxtun', 'Avonia', 'Mifflintown', 'Ballou', 'Annawan']
    # args.num_gpus = 1
    # args.process_split = [6]
    # scenes_per_process = 1

    # create vector env
    for gpu_id in range(args.num_gpus):
        for j in range(args.process_split[gpu_id]):
            # print(j)
            # print(scenes_per_process)
            config = get_config(args.config)
            # print(config)

            config.defrost()

            # config.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = int(4)
            config.DATASET.SPLIT = args.split
            config.DATASET.CONTENT_SCENES = scenes[scenes_per_process*j: (j+1)*scenes_per_process]

            # key
            config.DATASET.MAX_SCENE_REPEAT_STEPS = args.max_episode_length
            
            config.DATASET.DATA_PATH = absolute_path_prefix + config.DATASET.DATA_PATH
            config.DATASET.SCENES_DIR = absolute_path_prefix + config.DATASET.SCENES_DIR
           
            # key
            config.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = args.episodes_per_scene
            # config.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = args.episodes_per_scene
            # config.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
                # config.environment.iterator_options.max_scene_repeat_steps = args.pairs_per_scene
                # config.environment.iterator_options.cycle = False

            config.freeze()

            configs.append(config)
            args_list.append(args)

    envs = VectorEnv(
        make_env_fn=make_env_fn,
        env_fn_args=tuple(
            tuple(zip(args_list, configs, range(args.num_processes)))
        ),
        workers_ignore_signals=workers_ignore_signals,
    )
    return envs
