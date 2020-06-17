import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from modules import architecture
from dataloading import color_adjust
import utils
import random
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.nn.parallel
import torch.backends.cudnn as cudnn

# Training settings
parser = argparse.ArgumentParser(description="Color_adjust")
parser.add_argument("--batch_size", type=int, default=8,
                    help="training batch size")
parser.add_argument("--epochs", type=int, default=1200,
                    help="number of epochs to train")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning Rate. Default=2e-4")
parser.add_argument("--step_size", type=int, default=300,
                    help="learning rate decay per N epochs")
parser.add_argument("--gamma", type=int, default=0.5,
                    help="learning rate decay factor for step decay")
parser.add_argument("--start-epoch", default=1, type=int,
                    help="manual epoch number")
parser.add_argument("--workers", type=int, default=8,
                    help="number of threads for data loading")
parser.add_argument("--root", type=str, default=".",
                    help='dataset directory')
parser.add_argument("--n_train", type=int, default=1636,
                    help="number of training set")
parser.add_argument("--n_val", type=int, default=1,
                    help="number of validation set")
parser.add_argument("--rgb_range", type=int, default=1,
                    help="maxium value of RGB")
parser.add_argument("--n_colors", type=int, default=3,
                    help="number of color channels to use")
parser.add_argument("--pretrained", default="", type=str,
                    help="path to pretrained models")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--ext", type=str, default='.png')
parser.add_argument("--phase", type=str, default='train')
parser.add_argument("--world-size", default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument("--rank", default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument("--dist-url", default='tcp://127.0.0.1:10086', type=str,
                    help='url used to set up distributed training')
parser.add_argument("--dist-backend", default='nccl', type=str,
                    help='distributed backend')
parser.add_argument("--multiprocessing-distributed", action='store_true', default=True,
                    help='Use multi-processing distributed training to launch '
                         'N process per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_acc = 0.05

def main():
    args = parser.parse_args()
    print(args)
    # random seed
    seed = args.seed
    if seed is None:
        seed = random.randint(1, 10000)
    print("Ramdom Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    print("===> Building models")
    model = architecture.ColorNet()
    print_network(model)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.pretrained:

        if os.path.isfile(args.pretrained):
            print("===> loading models '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}

            for k, v in model_dict.items():
                if k not in pretrained_dict:
                    print(k)
            model.load_state_dict(pretrained_dict, strict=True)

        else:
            print("===> no models found at '{}'".format(args.pretrained))

    # define loss function and optimizer
    l1_criterion = nn.L1Loss().cuda(args.gpu)  # RGB
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    cudnn.benchmark = True

    # Data loading code

    train_dataset = color_adjust.color_adjust(args)  # 1636
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
    train_loader = DataLoader(dataset=train_dataset, num_workers=args.workers, batch_size=args.batch_size,
                              shuffle=(train_sampler is None), pin_memory=True, sampler=train_sampler, drop_last=True)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        utils.adjust_learning_rate(optimizer, epoch, args.step_size, args.lr, args.gamma)
        print('epoch =', epoch, 'lr = ', optimizer.param_groups[0]['lr'])

        acc = train(train_loader, model, l1_criterion, optimizer, epoch, args)
        # remeber best acc and save checkpoint
        is_best = acc < best_acc
        best_acc = min(acc, best_acc)
        if not args.multiprocessing_distributed or (
                args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_checkpoint(epoch, model)
            if is_best:
                save_best_checkpoint(model)


def train(train_loader, model, criterion1, optimizer, epoch, args):
    model.train()
    loss_total_l1= 0
    for iteration, (lr_tensor, hr_tensor) in enumerate(train_loader, 1):
        if args.gpu is not None:
            lr_tensor = lr_tensor.cuda(args.gpu, non_blocking=True)  # ranges from [0, 1]
        hr_tensor = hr_tensor.cuda(args.gpu, non_blocking=True)  # ranges from [0, 1]

        optimizer.zero_grad()
        sr_tensor = model(lr_tensor)
        loss_l1 = criterion1(sr_tensor, hr_tensor)
        loss_sr = loss_l1
        loss_total_l1 += loss_l1.item()
        loss_sr.backward()
        optimizer.step()
    print(
        "====> Epoch[{}]: Loss_mean_l1: {:.5f}".format(epoch, loss_total_l1 / len(train_loader)))
    return loss_total_l1 / len(train_loader)


def save_checkpoint(epoch, model):
    model_folder = "model/Color_adjust/"
    model_out_path = model_folder + "epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))

def save_best_checkpoint(model):
    model_folder = "model/Color_adjust/"
    model_out_path = model_folder + "best_model.pth"
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(), model_out_path)
    print("===> Checkpoint saved to {}".format(model_out_path))



def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


if __name__ == '__main__':
    main()
