import time
import shutil

import os
import os.path as osp

from dataset import construct_envs
from arguments import get_args

absolute_path_prefix = os.getcwd()

if __name__ == '__main__':
    args = get_args()

    output_dir = args.generate_dataset_output_dir.format(split=args.split)

    # if osp.exists(output_dir):
    #     sig = input(f'output_dir is not empty: {output_dir} \n'
    #                 f'if delete it ?[y|n]\t')
    #     if sig == 'y' or sig == 'Y':
    #         shutil.rmtree(output_dir)
    #         print("Successfully delete!")
    #     else:
    #         exit('Please change the output_dir!')

    assert not os.makedirs(output_dir, exist_ok=True), f'make dir [{output_dir}] failed!'

    envs = construct_envs(args)

    start = time.time()

    # run to generate dataset
    func_list = []
    for i in range(args.num_processes):
        func_list.append('run')
    envs.call(func_list)
    func_list = []

    # close env
    # for i in range(args.num_processes):
    #     func_list.append('close')
    # envs.call(func_list)

    end = time.time()
    print(f'{args.num_processes_per_gpu} gpus use time: {end - start}')
