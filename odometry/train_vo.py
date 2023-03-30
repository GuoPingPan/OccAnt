import os
import os.path as osp
import shutil
import time

from train.trainer import VoTrainer
from arguments import get_args

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #（代表仅使用第0，1号GPU）
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == '__main__':
    args = get_args()

    # if osp.exists(args.log_dir):
    #     sig = input(f'output_dir is not empty: {args.log_dir} \n'
    #                 f'if delete it ?[y|n]\t')
    #     if sig == 'y' or sig == 'Y':
    #         shutil.rmtree(args.log_dir)
    #         print("Successfully delete!")
    #     else:
    #         exit('Please change the output_dir!')
    try:
        shutil.rmtree(args.log_dir)
    except:
        pass

    assert not os.makedirs(args.log_dir, exist_ok=True), f'make dir [{args.log_dir}] failed!'

    trainer = VoTrainer(args)
    start = time.time()
    trainer.train()
    end = time.time()
    print(f"Training use time: {start - end}")