###常规的包###
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import random
###pytorch的包###
import torch
import torchvision
from torch.utils import data
from torch.optim import Adam, lr_scheduler, AdamW
from torch.utils.tensorboard import SummaryWriter
###自己定义的包###
from args_file import set_args
from dataset import ImageDataset, train_transform, test_transform
from network import initialize_weights
import pytorch_msssim
from network import HCFusion
from NIMA_loss import NIMA_Loss
from PDL_loss import PDL_loss
from Edge_loss import Edge_loss

# 固定随机种子
def seed_torch(seed=2022):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    # 设定随机种子
    seed_torch(seed=args.seed)

    # 获取计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-当前计算设备：{}".format(torch.cuda.get_device_name(0)))

    # 导入数据集
    train_set = ImageDataset(
        images_path=args.image_path,
        transform=train_transform,
        image_num=args.image_num
    )
    train_loader = data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    test_set = ImageDataset(
        images_path=args.image_path,
        transform=test_transform,
        image_num=16 * 10
    )
    test_loader = data.DataLoader(
        test_set,
        batch_size=16,
        num_workers=args.num_workers,
        shuffle=False
    )
    # 导入测试图像
    for i, batch in enumerate(test_loader):
        test_image = batch
        break
    test_image = test_image.to(device)
    print('-训练数据集与测试数据集导入完成')

    # 构建神经网络
    Train_network = HCFusion().to(device)
    print("-Train_network构建完成，参数量为： {} ".format(sum(x.numel() for x in Train_network.parameters())))

    # 损失函数和迭代器
    optimizer = AdamW(Train_network.parameters(), args.learning_rate, weight_decay=5E-2)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs // 3, eta_min=1e-6)
    mse_loss = torch.nn.MSELoss()
    ssim_loss = pytorch_msssim.msssim


    print('-损失函数及优化器构建完成')

    # 是否迁移学习
    start_epoch = 0
    if args.warm_start_path is not None:
        checkpoint = torch.load(args.warm_start_path)
        Train_network.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        Train_network.fusion_decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.param_groups[0]['lr'] = args.learning_rate
        print('--权重载入完成...')
    else:
        initialize_weights(Train_network)

    # 训练记录
    train_time = datetime.now().strftime("%m-%d_%H-%M")
    logs_name = train_time + '_epoch{}'.format(args.epochs + start_epoch)
    logs_dir = os.path.join('./logs/', logs_name)
    writer = SummaryWriter(logs_dir)
    print('-日志保存路径：' + logs_dir)
    print('--使用该指令查看训练过程：tensorboard --logdir=./')
    with open(os.path.join(logs_dir, 'info.txt'), 'a') as f:
        f.write(train_time + '\n')
        f.write('=-' * 30 + '\n')
        for arg in vars(args):
            f.write('--' + str(arg) + ':' + str(getattr(args, arg)) + '\n')
        f.write('=-' * 30 + '\n')

    # 保存权重的主路径
    save_path = os.path.join(args.checkpoint_path, logs_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 开始训练
    print('-开始训练...')
    step = 0  # 参数更新次数
    for epoch in range(start_epoch, args.epochs + start_epoch):
        loop = tqdm(train_loader)
        for _, image_batch in enumerate(loop):
            # 载入图像
            image_batch = image_batch.to(device)
            target = image_batch.data.clone().to(device)
            outputs = Train_network(image_batch)

            optimizer.zero_grad()

            pixel_loss_value = mse_loss(outputs, target)
            ssim_loss_value = 1 - ssim_loss(outputs, target, normalize=True)

            total_loss_value = pixel_loss_value + args.ssim_weight * ssim_loss_value

            total_loss_value.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch + 1}/{args.epochs + start_epoch}]")
            loop.set_postfix(
                pixel_loss=pixel_loss_value.item(),
                ssim_loss=ssim_loss_value.item(),
                total_loss=total_loss_value.item()
            )

            step += 1
            # 测试图像重建结果
            if step % args.tensorboard_step == 1:
                with torch.no_grad():
                    writer.add_scalar('pixel_loss', pixel_loss_value.item(), global_step=step)
                    writer.add_scalar('ssim_loss', ssim_loss_value.item(), global_step=step)
                    writer.add_scalar('total_loss', total_loss_value.item(), global_step=step)
                    # Train_network.eval()
                    rebuild_img = Train_network(test_image)
                    img_grid_real = torchvision.utils.make_grid(
                        test_image, normalize=False, nrow=4
                    )
                    img_grid_rebuild = torchvision.utils.make_grid(
                        rebuild_img, normalize=False, nrow=4
                    )

                    writer.add_image('Real image', img_grid_real, global_step=1)
                    writer.add_image('Rebuild image', img_grid_rebuild, global_step=step)
                    # Train_network.train()

            if step <= 100:
                with torch.no_grad():
                    # Train_network.eval()
                    # for i, test_bench in enumerate(test_loader):
                    #     if i == 0:
                    #         test_bench = test_bench.to(device)
                    #         rebuild_img_100 = Train_network(test_bench)
                    #         img_grid_rebuild = torchvision.utils.make_grid(
                    #             rebuild_img_100, normalize=False, nrow=4
                    #         )
                    #         writer.add_image('Rebuild image 100', img_grid_rebuild, global_step=step)
                    #     else:
                    #         break
                    rebuild_img_100 = Train_network(test_image)
                    img_grid_rebuild = torchvision.utils.make_grid(
                        rebuild_img_100, normalize=False, nrow=4
                    )

                    writer.add_image('Rebuild image 100', img_grid_rebuild, global_step=step)
                    # print(rebuild_img_100.max(), rebuild_img_100.min())

                    # Train_network.train()

        # 学习率记录
        # writer.add_scalar('cnn_lr', optimizer.param_groups[0]['lr'], global_step=epoch + 1)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch + 1)
        scheduler.step()

        # 保存权重
        save_name = os.path.join(save_path + '/',
                                 'epoch{}_{}.pt'.format(args.epochs + start_epoch, epoch + 1))
        torch.save(
            {
                'encoder_state_dict': Train_network.CNN.state_dict(),
                'decoder_state_dict': Train_network.Trans.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': (args.epochs + start_epoch)
            }, save_name
        )
        print('模型数据已保存在：' + save_name)

    print('-训练完成')


if __name__ == "__main__":
    args = set_args()
    main(args)
