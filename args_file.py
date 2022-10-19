"""
file - args_file.py
用于设定所有参数
"""

import argparse


def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--code_writer', type=str, default="WJQ & YY", help="Name of code writer")
    # 数据相关参数
    parser.add_argument('--image_path', default=r'F:\\training_ava\\test\\', type=str, help='训练集路径')
    parser.add_argument('--image_num', default=80000, type=int, help='用于训练的图像数量')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader使用的cpu线程数')
    parser.add_argument('--batch_size', type=int, default=10, help='批量大小')

    # 训练相关参数
    parser.add_argument('--seed', type=int, default=12, help='随机种子')
    parser.add_argument('--epochs', type=int, default=8, help='训练周期数')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='学习率')
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint/')
    parser.add_argument('--warm_start_path', type=str,
                        # 'D:/MyFusion/NewFusion/checkpoint/07-02_00-21_epoch4/epoch4_4.pt',
                        default=None,
                        help='继续训练的路径')
    parser.add_argument('--tensorboard_step', default=25, type=int, help='tensorboard的更新步数')

    parser.add_argument('--ssim_weight', type=float, default=10.0, help='ssim损失的权重')


    # 显示参数
    args = parser.parse_args()
    print('=-' * 30)
    for arg in vars(args):
        print('--', arg, ':', getattr(args, arg))
    print('=-' * 30)

    return args


if __name__ == '__main__':
    args = set_args()
