import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, BatchSampler

from models.vo import VoNet
from dataset.dataloader import VoDataset
from utils import transform, metric
from loss import *

import os
import os.path as osp
import logging
logger = logging.getLogger("train_logger")


np.set_printoptions(precision=6)

from torch.utils.tensorboard import SummaryWriter

class VoTrainer:
    '''
        Trainer 需要完成的工作
        1.四个encoder: pose、pose_uncertainty(optional)
        2.loss构建：
            pose：MAE、Uncetainty(optional)
        3.DataLoader
        4.支持并行训练
    '''
    def __init__(self, args):
        super(VoTrainer, self).__init__()

        self.args = args
        self.log_dir = args.log_dir
        self.save_model_epochs = self.args.save_model_epochs
        self.decoder_type = args.decoder_type
        # build model
        self.model = VoNet(num_layers=args.num_layers,
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

        # for k in self.model.parameters():
        #     print(k)
        # exit()

        self.device = args.device
        if args.vonet_checkpoint is not None:
            assert osp.exists(args.vonet_checkpoint), f"checkpoint dosen't exit: {args.vonet_checkpoint}"
            self.model.load_state_dict(torch.load(args.vonet_checkpoint))
            print("successfully load!")

        if args.save_model:
            assert not os.makedirs(osp.join(args.log_dir, 'checkpoints'), exist_ok=True), \
                f"make dir [{osp.join(args.log_dir, 'checkpoints')}] failed!"

        # build optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        # self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer,
        #     args.lr, args.lr_step)

        self.init_loss()

        print(f"Can use device nums: {torch.cuda.device_count()}")

        # build dataloader
        train_dataset = VoDataset(data_dir=args.dataset, split=args.split, add_obs_noise=args.add_obs_noise)
        val_dataset = VoDataset(data_dir=args.dataset, split="val", add_obs_noise=args.add_obs_noise)

        nw = min([os.cpu_count(), args.batch_size, 8])
        print(f"All CPU: {os.cpu_count()}")
        print(f"Use CPU: {nw}")
        nw = 8 # 为 0的时候直接用满
        print(f"batch size: {args.batch_size}")
        if self.device == "cpu":
            self.gpus = 0
            self.train_dataloader = DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=nw,
                                            shuffle=True,
                                            collate_fn=VoDataset.collate_fn
                                            )
            self.val_dataloader = DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=nw,
                                            shuffle=True,
                                            collate_fn=VoDataset.collate_fn
                                            )
            print("Use CPU!")

        elif torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.fastest = True

            # 单卡
            if torch.cuda.device_count() == 1:
                self.gpus = 1
                self.model.to(self.device)
                self.train_dataloader = DataLoader(train_dataset,
                                            batch_size=args.batch_size,
                                            pin_memory=True,  # 将数据加载到gpu中
                                            num_workers=nw,
                                            shuffle=True,
                                            collate_fn=VoDataset.collate_fn
                                            )
                self.val_dataloader = DataLoader(val_dataset,
                                            batch_size=args.batch_size,
                                            pin_memory=True,  # 将数据加载到gpu中
                                            num_workers=nw,
                                            shuffle=True,
                                            collate_fn=VoDataset.collate_fn
                                            )
                print("Use single GPU!")

            elif torch.cuda.device_count() > 1:
                self.init_distributed_mode(args)
                self.rank = args.rank
                self.world_size = args.world_size
                self.lr = args.lr * self.world_size  # 调账学习率步长，因为梯度为多个向量的均值
                self.gpus = torch.cuda.device_count()

                self.train_sampler = DistributedSampler(train_dataset, rank=self.rank, shuffle=True)
                self.val_sampler = DistributedSampler(val_dataset, rank=self.rank, shuffle=True)
                train_batch_sampler = BatchSampler(self.train_sampler, batch_size=args.batch_size, drop_last=True)
                val_batch_sampler = BatchSampler(self.val_sampler, batch_size=args.batch_size, drop_last=True)
                self.train_dataloader = DataLoader(train_dataset,
                                                pin_memory=True,  # 将数据加载到gpu中
                                                num_workers=nw,
                                                batch_sampler=train_batch_sampler,
                                                collate_fn=VoDataset.collate_fn
                                                )
                self.val_dataloader = DataLoader(val_dataset,
                                                pin_memory=True,  # 将数据加载到gpu中
                                                num_workers=nw,
                                                sampler=val_batch_sampler,
                                                collate_fn=VoDataset.collate_fn
                                                )
                # 这里device默认是cuda,因为在init进程的时候已经创建set_device了,因此这里会自动分配gpu
                self.model.to(self.device)

                if args.vonet_checkpoint is None:
                    # 由于这里要保证每个模型的初始权重相同,因此要先保存再重新加载
                    if self.rank == 0:
                        ckpt_path = os.path.join(self.log_dir, "checkpoints/initial.ckpt")
                        if not os.path.exists(ckpt_path):
                            torch.save(self.model.state_dict(), ckpt_path)
                
                    dist.barrier()
                    state_dict = torch.load(ckpt_path, map_location=self.device)
                    self.model.load_state_dict(state_dict, strict=True)
                    print("successfully load!")

                # 这里的args.gpu应该是当前进程使用的gpu,而上面的device是"cuda"
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
                print("Use multiple GPU!")

    def init_logger(self, path):
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(path)
        logger.addHandler(handler)

    def init_loss(self):
        self.pose_l1_loss = PoseL1Loss()
        self.pose_laplacian_loss = PoseLaplacianUncerLoss()
        self.pose_nig_loss = PoseNIGLoss()
        self.pose_gaussian_loss = PoseGaussianLoss()

    def train_one_epoch(self):

        torch.cuda.empty_cache()
        self.optimizer.zero_grad()
        self.model.train()

        batches = len(self.train_dataloader)

        metric_dict = {
            "move_forward_trans_mae": torch.zeros(1, dtype=torch.float),
            "turn_left_trans_mae": torch.zeros(1, dtype=torch.float),
            "turn_right_trans_mae": torch.zeros(1, dtype=torch.float),

            "move_forward_rot_mae": torch.zeros(1, dtype=torch.float),
            "turn_left_rot_mae": torch.zeros(1, dtype=torch.float),
            "turn_right_rot_mae": torch.zeros(1, dtype=torch.float),

            "trans_uncer": torch.zeros(3, dtype=torch.float),
            "trans_uncer_mean": torch.zeros(1, dtype=torch.float),
            "rot_uncer_mean": torch.zeros(1, dtype=torch.float),

            "loss": torch.zeros(1, dtype=torch.float)
        }

        for batch, data in enumerate(self.train_dataloader):
            action = data["action_id"]

            move_forward_index = (action == 1)
            turn_left_index = (action == 2)
            turn_right_index = (action == 3)
            
            index_masks = {
                "move_forward": move_forward_index, 
                "turn_left": turn_left_index, 
                "turn_right": turn_right_index
                }
            
            pose_loss_reduce = 0.0
            reg_loss_reduce = 0.0
            
            trans_uncer_aa = torch.zeros(3, dtype=torch.float)
            trans_uncer_aa_mean = torch.zeros(1, dtype=torch.float)
            rot_uncer_aa = torch.zeros(1, dtype=torch.float)

            for k, mask_index in index_masks.items():
                if mask_index.sum() != 0:

                    factor = mask_index.sum() / self.args.batch_size

                    rgb_t_1 = data["rgb_t_1"][mask_index]
                    rgb_t = data["rgb_t"][mask_index]
                    rgb_input = torch.concat([rgb_t_1, rgb_t], dim=1).to(self.device)

                    action = data["action_id"][mask_index].to(self.device)
                    collision = data["collision"][mask_index].to(self.device)
                    pose_delta_gt = data["pose_delta"][mask_index].to(self.device)

                    out = self.model(rgb_input, action, collision)

                    trans_mae, rot_mae = metric.rot_and_trans_mae(out, pose_delta_gt)
                    metric_dict[k + "_trans_mae"] += trans_mae * factor
                    metric_dict[k + "_rot_mae"] += rot_mae * factor

                    if self.decoder_type == "base":
                        pose_loss = self.pose_l1_loss(out, pose_delta_gt)
                    elif self.decoder_type == "laplacian":
                        pose_loss = self.pose_l1_loss(out, pose_delta_gt) + self.pose_laplacian_loss(out)
                        trans_uncer, rot_uncer = metric.rot_and_trans_laplacian_uncer(out)
                        trans_uncer_aa += trans_uncer*factor
                        trans_uncer_aa_mean += torch.mean(trans_uncer)*factor
                        rot_uncer_aa += rot_uncer*factor
                    elif self.decoder_type == "gaussian":
                       pose_loss = self.pose_l1_loss(out, pose_delta_gt) + self.pose_gaussian_loss(out)
                    elif self.decoder_type == "nig":
                        pose_loss, reg_loss = self.pose_nig_loss(out, pose_delta_gt)
                        reg_loss_reduce += reg_loss.clone().detach().cpu() * factor
                        trans_uncer, rot_uncer = metric.rot_and_trans_nig_uncer(out)
                        trans_uncer_aa += trans_uncer*factor
                        trans_uncer_aa_mean += torch.mean(trans_uncer)*factor
                        rot_uncer_aa += rot_uncer*factor
                    else:
                        raise NotImplementedError

                    pose_loss_reduce += pose_loss * factor

            # print(pose_loss_reduce)
            pose_loss_reduce.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.decoder_type == "nig":
                self.pose_nig_loss.lam += self.pose_nig_loss.maxi_rate*(reg_loss_reduce - self.pose_nig_loss.epsilon)

            # logging
            if self.gpus <= 1 or self.rank == 0:
                if (self.gpus > 1 and self.rank == 0):
                    metric_dict["loss"] += self.reduce_value(pose_loss_reduce.detach(), average=True)
                    for k, v in metric_dict.items():
                        if "mae" in k:
                            metric_dict[k] = self.reduce_value(v, average=True)
                    metric_dict["trans_uncer"] += self.reduce_value(trans_uncer_aa, average=True)
                    metric_dict["trans_uncer_mean"] += self.reduce_value(trans_uncer_aa_mean, average=True)
                    metric_dict["rot_uncer_mean"] += self.reduce_value(rot_uncer_aa, average=True)
                else:
                    metric_dict["loss"] += pose_loss_reduce.detach().cpu()
                    metric_dict["trans_uncer"] += trans_uncer_aa
                    metric_dict["trans_uncer_mean"] += trans_uncer_aa_mean
                    metric_dict["rot_uncer_mean"] += rot_uncer_aa

                if batch % self.args.vonet_log_interval == 0:

                    log = f"[Trainning] Batch:[{batch:>3d}/{batches:>3d}]\n" 
                    for k, v in metric_dict.items():
                        # print(k, v, batch)
                        if k == "trans_uncer": 
                            log += f"\t {k}: {v.numpy()/(batch+1)}\n"
                            continue
                        log += f"\t {k}: {v.item()/(batch+1):.6f}\n"

                    print(log)
                    logger.info(log)

        for k, v in metric_dict.items():
            if k == "trans_uncer":
                metric_dict[k] = v.numpy() / batches
                continue
            metric_dict[k] = v.item() / batches 

        # 等待所有进程计算完毕
        if self.gpus > 1:
            torch.cuda.synchronize(self.device)

        return metric_dict

    def test_one_epoch(self):

        log = f"------------------------- Evaluation ------------------------- "
        print(log)
        logger.info(log)

        torch.cuda.empty_cache()
        self.model.eval()
       
        metric_dict = {
            "move_forward_trans_mae": torch.zeros(1, dtype=torch.float),
            "turn_left_trans_mae": torch.zeros(1, dtype=torch.float),
            "turn_right_trans_mae": torch.zeros(1, dtype=torch.float),

            "move_forward_rot_mae": torch.zeros(1, dtype=torch.float),
            "turn_left_rot_mae": torch.zeros(1, dtype=torch.float),
            "turn_right_rot_mae": torch.zeros(1, dtype=torch.float),

            "trans_uncer": torch.zeros(3, dtype=torch.float),
            "trans_uncer_mean": torch.zeros(1, dtype=torch.float),
            "rot_uncer_mean": torch.zeros(1, dtype=torch.float),

            "loss": torch.zeros(1, dtype=torch.float)
        }

        batches = len(self.val_dataloader)
        with torch.no_grad():
            for batch, data in enumerate(self.val_dataloader):

                action = data["action_id"]

                move_forward_index = (action == 1)
                turn_left_index = (action == 2)
                turn_right_index = (action == 3)

                index_masks = {
                    "move_forward": move_forward_index, 
                    "turn_left": turn_left_index, 
                    "turn_right": turn_right_index}
                
                pose_loss_reduce = 0.0
                reg_loss_reduce = 0.0


                trans_uncer_aa = torch.zeros(3, dtype=torch.float)
                trans_uncer_aa_mean = torch.zeros(1, dtype=torch.float)
                rot_uncer_aa = torch.zeros(1, dtype=torch.float)

                for k, mask_index in index_masks.items():
                    if mask_index.sum() != 0:

                        factor = mask_index.sum() / self.args.batch_size

                        rgb_t_1 = data["rgb_t_1"][mask_index]
                        rgb_t = data["rgb_t"][mask_index]
                        rgb_input = torch.concat([rgb_t_1, rgb_t], dim=1).to(self.device)

                        action = data["action_id"][mask_index].to(self.device)
                        collision = data["collision"][mask_index].to(self.device)
                        pose_delta_gt = data["pose_delta"][mask_index].to(self.device)

                        out = self.model(rgb_input, action, collision)

                        trans_mae, rot_mae = metric.rot_and_trans_mae(out, pose_delta_gt)
                        metric_dict[k + "_trans_mae"] += trans_mae * factor
                        metric_dict[k + "_rot_mae"] += rot_mae * factor

                        if self.decoder_type == "base":
                            pose_loss = self.pose_l1_loss(out, pose_delta_gt)
                        elif self.decoder_type == "laplacian":
                            pose_loss = self.pose_l1_loss(out, pose_delta_gt) + self.pose_laplacian_loss(out)
                            trans_uncer, rot_uncer = metric.rot_and_trans_laplacian_uncer(out)
                            trans_uncer_aa += trans_uncer*factor
                            trans_uncer_aa_mean += torch.mean(trans_uncer)*factor
                            rot_uncer_aa += rot_uncer*factor
                        elif self.decoder_type == "gaussian":
                            pose_loss = self.pose_l1_loss(out, pose_delta_gt) + self.pose_gaussian_loss(out)
                        elif self.decoder_type == "nig":
                            pose_loss, reg_loss = self.pose_nig_loss(out, pose_delta_gt)
                            reg_loss_reduce += reg_loss.clone().detach().cpu() * factor
                            trans_uncer, rot_uncer = metric.rot_and_trans_nig_uncer(out)
                            trans_uncer_aa += trans_uncer*factor
                            trans_uncer_aa_mean += torch.mean(trans_uncer)*factor
                            rot_uncer_aa += rot_uncer*factor
                        else:
                            raise NotImplementedError

                        pose_loss_reduce += pose_loss * factor


                # logging
                if (self.gpus <= 1 or self.rank == 0) and batch % 100 == 0:
                    if (self.gpus > 1):
                        metric_dict["loss"] += self.reduce_value(pose_loss_reduce.detach(), average=True)
                        for k, v in metric_dict.items():
                            if "mae" in k:
                                metric_dict[k] = self.reduce_value(v, average=True)
                        metric_dict["trans_uncer"] += self.reduce_value(trans_uncer_aa, average=True)
                        metric_dict["trans_uncer_mean"] += self.reduce_value(trans_uncer_aa_mean, average=True)
                        metric_dict["rot_uncer_mean"] += self.reduce_value(rot_uncer_aa, average=True)
                    else:
                        metric_dict["loss"] += pose_loss_reduce.detach().cpu()
                        metric_dict["trans_uncer"] += trans_uncer_aa
                        metric_dict["trans_uncer_mean"] += trans_uncer_aa_mean
                        metric_dict["rot_uncer_mean"] += rot_uncer_aa
    

                    log = f"[Evaluation] Batch:[{batch:>3d}/{batches:>3d}]\n" 
                    for k, v in metric_dict.items():
                        if k == "trans_uncer": 
                            log += f"\t {k}: {(v.numpy()/(batch+1))}\n"
                            continue
                        log += f"\t {k}: {v.item()/(batch+1):.6f}\n"

                    print(log)
                    logger.info(log)

      
            for k, v in metric_dict.items():
                if k == "trans_uncer":
                    metric_dict[k] = v.numpy() / batches
                    continue
                metric_dict[k] = v.item() / batches 


            # 等待所有进程计算完毕
            if self.gpus > 1:
                torch.cuda.synchronize(self.device)

        return metric_dict, self.model.state_dict()

    def train(self):
        min_loss = 1e6
        best_model = None
        best_epoch = None

        if self.gpus <= 1 or self.rank == 0:
            if self.args.save_model:
                os.makedirs(self.log_dir, exist_ok=True)


            self.init_logger(path=os.path.join(self.log_dir, "train_log.txt"))
            writer = SummaryWriter(log_dir=os.path.join(self.log_dir, "tb"))
            metric_logger = logging.getLogger("metric_logger")
            metric_logger.setLevel(logging.INFO)
            handler = logging.FileHandler(os.path.join(self.log_dir, "metric_log.txt"))
            metric_logger.addHandler(handler)   

        for epoch in range(self.args.epochs):
            if self.gpus > 1:
                # 这里会根据每个epoch生成不同的种子来打乱数据
                self.train_sampler.set_epoch(epoch)

            if self.gpus <= 1 or self.rank == 0:
                log = f"[Epoch]:[{epoch + 1:>3d}/{self.args.epochs:>3d}]\n"
                print(log)
                metric_logger.info(log)

            metric_dict = self.train_one_epoch()
            
            if self.gpus <= 1 or self.rank == 0:
                log = f"[Epoch Train]:[{epoch + 1:>3d}/{self.args.epochs:>3d}]\n"

                for k, v in metric_dict.items():
                    if k == "trans_uncer": 
                        log += f"\t {k}: {v.tolist()}\n"
                        continue
                    writer.add_scalar(f"Train/{k}", v, epoch)
                    log += f"\t {k}: {v:.6f}\n"

                print(log)
                metric_logger.info(log)


            metric_dict, model_dict = self.test_one_epoch()

            if metric_dict["loss"] < min_loss:
                min_loss = metric_dict["loss"]
                best_model = model_dict
                best_epoch = epoch

            if self.gpus <= 1 or self.rank == 0:
                log = f"[Epoch Eval]:[{epoch + 1:>3d}/{self.args.epochs:>3d}]\n" 

                for k, v in metric_dict.items():
                    if k == "trans_uncer": 
                        log += f"\t {k}: {v.tolist()}\n"
                        continue
                    writer.add_scalar(f"Test/{k}", v, epoch)
                    log += f"\t {k}: {v:.6f}\n"

                print(log)
                metric_logger.info(log)


                if self.save_model_epochs and epoch % self.save_model_epochs == 0:
                    if self.args.save_model:
                        self.save_checkpoint(model_dict, f"checkpoints/ep{int(epoch)}.pth")

        if self.gpus <= 1 or self.rank == 0:
            if self.args.save_model:
                self.save_checkpoint(best_model, f"checkpoints/ep{int(best_epoch)}_best.pth")

        if self.gpus > 1:
            self.cleanup()
  
    def save_checkpoint(self, state_dict, shuffix=""):
        torch.save(state_dict, osp.join(self.log_dir, shuffix))

    def reduce_value(self, value, average=True):

        with torch.no_grad():
            dist.all_reduce(value)
            if average:
                value /= self.world_size

            return value

    def cleanup(self):
        dist.destroy_process_group()

    def init_distributed_mode(self, args):
        
        assert dist.is_available(), "distributed training is not available"
        
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            # 单机多卡的时候 RANK = LOCAL_RANK
            # 多机多卡的时候 RANK 代表 全局下的第几个进程
            #             LOCAL 代表 当前机器下的第几个进程
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.gpu = int(os.environ["LOCAL_RANK"])

        elif "SLURM_PROCID" in os.environ:
            args.rank = int(os.environ["SLURM_PROCID"])
            args.gpu = args.rank % torch.cuda.device_count()
        else:
            print("Not using distributed mode")
            args.distributed = False
            return

        args.distributed = True

        torch.cuda.set_device(args.gpu)
        args.dist_backend = "nccl"  # 通信后端，nvidia GPU推荐使用NCCL
        args.init_method = "env://"
        print("| Distributed init (rank {}): {}".format(args.rank, "env://"), flush=True)
        print(f"world_size: {args.world_size}")
        # 多个进程的world_size一样,但是rank不一样
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.init_method,
                                world_size=args.world_size, 
                                rank=args.rank)
        dist.barrier()
        print("Successfully init process group")
    