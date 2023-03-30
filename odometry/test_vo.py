import time
import matplotlib.pyplot as plt
import sys
import shutil
sys.path.append('/home/hello/pgp_eai/easan')

import os
import os.path as osp
os.environ['HABITAT_SIM_LOG'] = "quiet"
os.environ['MAGNUM_LOG'] = "quiet"

import torch
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from odometry import VoNet
from odometry.utils import metric, transform

from dataset.dataloader import VoDataset
from arguments import get_args

# os.environ['CUDA_VISIBLE_DEVICES'] = "0"  #（代表仅使用第0，1号GPU）
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 





def display_sample(rgb_obs_1, rgb_obs, depth_est, depth_gt, depth_uncer_est, log_pth):


    rgb_1_img = Image.fromarray(rgb_obs_1, mode='RGB')

    rgb_img = Image.fromarray(rgb_obs, mode='RGB')


    depth_img = Image.fromarray((depth_est / 10. * 255).astype(np.uint8), mode="L")

    depth_gt_img = Image.fromarray((depth_gt / 10. * 255).astype(np.uint8), mode="L")

    depth_delta = np.abs(depth_est - depth_gt)
    depth_delta_img = Image.fromarray((depth_delta / 10. * 255).astype(np.uint8), mode="L")

    # depth_uncer_img = Image.fromarray((depth_uncer * 255).astype(np.uint8), mode="L")

    fig, axs = plt.subplots(figsize=(12, 8), nrows=2, ncols=3)
    fig.subplots_adjust(wspace=0.5)


    imgs = [rgb_img, depth_img, depth_delta_img, rgb_1_img, depth_gt_img, depth_uncer_est]
    titles = ['rgb', 'depth_est', 'depth_delta', '$rgb_1$', 'depth_gt', 'depth_uncer']
    cmaps = [None, 'magma', 'hot', None, 'magma', 'hot']

    pos_temp = []


    mae_mean, _ = metric.mae_mean_max_numpy(depth_est, depth_gt)
    sigma_ = metric.sigma_numpy(depth_est, depth_gt)
    fig.text(
        0.5,
        0.92,
        f"depth_mae: {mae_mean:.6f}\nsigma: {sigma_:.6f}",
        size=12, family="serif", color="blue", style='italic', weight="light",
        bbox=dict(facecolor="dodgerblue", alpha=0.5, boxstyle="round")
    )

    fontdict = {'family': 'serif', 'style':'italic'}

    for i, ax in enumerate((axs).reshape(-1)):
        if imgs[i] is None:
            pos_temp.append(None)
            ax.axis('off')
            continue
        pos_temp.append(ax.imshow(imgs[i], cmap=cmaps[i]))
        ax.set_title(titles[i], size=12, fontdict=fontdict, color="blue", weight="light",)
        ax.axis('off')



    # # delta colorbar
    # cax = fig.add_axes([
    #     axs[0, 2].get_position().x1 + 0.02,
    #     axs[0, 2].get_position().y0,
    #     0.02,
    #     axs[0, 2].get_position().height
    # ])
    # fig.colorbar(pos_temp[2], ax=axs[0, 2], cax=cax)

    # # uncertainty colorbar
    # cax = fig.add_axes([
    #     axs[1, 2].get_position().x1 + 0.02,
    #     axs[1, 2].get_position().y0,
    #     0.02,
    #     axs[1, 2].get_position().height,
    # ])
    # fig.colorbar(pos_temp[5], ax=axs[1, 2], cax=cax)


    # # depth colorbar
    # cax = fig.add_axes([
    #     axs[1, 1].get_position().x1 + 0.02,
    #     0.2,
    #     0.02,
    #     0.6
    # ])
    # fig.colorbar(pos_temp[4], ax=axs[:, 1], cax=cax)


    plt.tight_layout()

    plt.savefig(log_pth)
    print(f'Successfully save image to: {log_pth}')
    # plt.show()


if __name__ == '__main__':
    args = get_args()

    log_dir = args.test_depth_out_dir

    if log_dir:
        if osp.exists(log_dir):
            sig = input(f'output_dir is not empty: {log_dir} \n'
                        f'if delete it ?[y|n]\t')
            if sig == 'y' or sig == 'Y':
                shutil.rmtree(log_dir)
                print("Successfully delete!")
            else:
                exit('Please change the output_dir!')

        shutil.rmtree(log_dir)

        assert not os.makedirs(log_dir, exist_ok=True), f'make dir [{log_dir}] failed!'


    assert not os.makedirs(args.test_depth_out_dir, exist_ok=True), f'make dir [{args.test_depth_out_dir}] failed!'


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
        print('Successfully load!')
        model.load_state_dict(torch.load(args.vonet_checkpoint))

    train_dataset = VoDataset(data_dir=args.dataset, split=args.split)

    test_dataloader = DataLoader(train_dataset,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=VoDataset.collate_fn
                                )

    model.to(args.device)
    model.eval()




    batches = len(test_dataloader)
    mae_mean_all = 0.0
    sigma_all = 0.0

    a = np.random.choice(batches-1, size=args.num_test_pic).tolist()


    ''' test all picture'''
    for i, data in enumerate(test_dataloader):

        with torch.no_grad():
            rgb_t_1 = data['rgb_t_1']
            rgb_t = data['rgb_t']
            rgb_input = torch.concat([rgb_t_1, rgb_t], dim=1).to(args.device)

            action = data['action_id'].to(args.device).to(args.device)
            depth_gt = data['depth_t'].to(args.device).to(args.device)
            pose_delta_gt = data['pose_delta'].to(args.device).to(args.device)
            collision = data['collision'].to(args.device)

            start = time.time()
            # pose_delta_est, pose_uncer_est, disp_est, depth_uncer_est = model(rgb_input, action)
            pose_nig_est, disp_est, depth_uncer_est = model(rgb_input, action, collision)
            # pose_avar = 100.*(pose_nig_est['beta'] / ((pose_nig_est['alpha'] - 1))).abs().detach().sum()
            # pose_evar = 100.*(pose_nig_est['beta'] / (pose_nig_est['v'] * (pose_nig_est['alpha'] - 1))).abs().detach().sum()

            # a_delta = np.sum(np.abs(pose_delta_gt.cpu().numpy() - actions[data['action_id'].cpu().item()]))
            # e_delta = np.sum((pose_nig_est['mu']-pose_delta_gt).abs().cpu().numpy())
            # if delta > thres:
            #     print('error')

            # pose_error += (pose_nig_est['mu']-pose_delta_gt).abs().sum()
            # pose_error += (pose_delta_est-pose_delta_gt).abs().sum()


            end = time.time()
            # print(f'use time: {end - start}         fps: {1 / (end - start)}')
            #
            # print(f"pose_delta_est: {pose_nig_est['mu'].cpu().numpy().tolist()},     "
            #       f"\npose_delta_gt: {pose_delta_gt.cpu().numpy().tolist()} noiseless: {actions[data['action_id'].cpu().item()]}"
            #       f"\ndelta: {(pose_nig_est['mu']-pose_delta_gt).cpu().numpy().tolist()},    abs:{(pose_nig_est['mu']-pose_delta_gt).abs().sum():.6f},"
            #       # f"    uncer_det:{pose_var.cpu().numpy().tolist()}"
            #       )
            # print(f'pose_uncer_est: {pose_uncer_est.data}')

            # print(f"a_delta: {a_delta}, pose_avar: {pose_avar}")
            # print(f"e_delta: {e_delta}, pose_evar: {pose_evar}")


            depth_est = transform.disp_to_uniform_depth(disp_est)

            rgb_obs_1 = (rgb_t_1[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
            rgb_obs = (rgb_t[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
            depth_est = np.squeeze(depth_est[0].permute(1, 2, 0).cpu().detach().numpy())
            depth_gt = np.squeeze(depth_gt[0].permute(1, 2, 0).cpu().detach().numpy())
            depth_uncer_est = np.squeeze(depth_uncer_est[0].permute(1, 2, 0).cpu().detach().numpy())

            mae_mean, mae_max = metric.mae_mean_max_numpy(depth_est, depth_gt)
            # print("mae_max: ", mae_max)
            mae_mean_all += mae_mean
            sigma_ = metric.sigma_numpy(depth_est, depth_gt)
            sigma_all += sigma_

            log_pth = osp.join(args.test_depth_out_dir, f'{i}.png')

            if i in a:
                display_sample(rgb_obs_1, rgb_obs, depth_est, depth_gt, depth_uncer_est, log_pth)
    print(f"mae_mean_all: {mae_mean_all / batches}    sigma_all: {sigma_all / batches}")