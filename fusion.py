import time
import os
import argparse
from PIL import Image
import torch
from torchvision.utils import save_image
from torchvision import transforms as transforms
from network import Can_Encoder, ConvNeXt_Encoder, Fuion_Decoder

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

encoder_dict = {
    'can': Can_Encoder(),
    'convnext': ConvNeXt_Encoder()
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--IR_image_path', type=str, default='', help='modal1')
    parser.add_argument('--VI_image_path', type=str, default='', help='modal2')
    parser.add_argument('--Fusion_image_path', type=str, default='', help='融合图片的保存路径')
    parser.add_argument('--encoder_type', type=str, default='can', help='encoder类型，可选：can、convnext')
    parser.add_argument('--checkpoint_path', type=str,
                        default='',
                        help='权重保存路径')
    args = parser.parse_args()
    return args

def main(args):
    # 构建保存路径
    if not os.path.exists(args.Fusion_image_path):
        os.makedirs(args.Fusion_image_path)

    # 获取计算设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("-当前计算设备：{}".format(torch.cuda.get_device_name(0)))

    # 构建网络
    Encoder = encoder_dict[args.encoder_type].to(device)
    Decoder = Fuion_Decoder().to(device)
    print('-网络构建完成...')

    # 载入权重
    checkpoint = torch.load(args.checkpoint_path)
    Encoder.load_state_dict(checkpoint['encoder_state_dict'])
    Decoder.load_state_dict(checkpoint['decoder_state_dict'])
    Encoder.eval()
    Decoder.eval()
    print('-权重载入完成')

    # 获取图像的文件名列表
    image_list = os.listdir(args.IR_image_path)

    # 用于张量转换
    fusion_transform = transforms.ToTensor()

    print('开始融合...')
    for image_name in image_list:
        time_start = time.time()  # 记录开始时间

        VI_image_path = os.path.join(args.VI_image_path, image_name)
        IR_image_path = os.path.join(args.IR_image_path, image_name)


       # save_path = os.path.join(args.Fusion_image_path, image_name)

        basename, houzhui = os.path.splitext(image_name)
        save_path = os.path.join(args.Fusion_image_path, basename + '_convnext_adapt_p0.0001' + '.jpg')


        # 读取图片
        VI_image = Image.open(VI_image_path).convert('RGB')
        IR_image = Image.open(IR_image_path).convert('RGB')
        VI_image = fusion_transform(VI_image).unsqueeze(0).to(device)
        IR_image = fusion_transform(IR_image).unsqueeze(0).to(device)

        with torch.no_grad():
            # 编码
            VI_image_EN = Encoder(VI_image)
            IR_image_EN = Encoder(IR_image)

            # 融合解码
            Fusion_image = Decoder(IR_image_EN, VI_image_EN)

        # 融合张量处理
        Fusion_image = Fusion_image.squeeze(0).detach().cpu()
        save_image(Fusion_image, save_path)

        time_end = time.time()  # 记录结束时间
        print('输出路径：' + save_path + '-融合耗时：{}'.format(time_end - time_start))


if __name__ == '__main__':
    args = get_args()
    main(args)
