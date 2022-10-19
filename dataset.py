import os
from PIL import Image
import torch
from torch.utils import data
from torchvision import transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2


class ImageDataset(data.Dataset):
    def __init__(self, images_path, transform=None, image_num=None):
        self.images_path = images_path
        self.image_list = os.listdir(images_path)
        self.transform = transform
        if image_num is not None:
            self.image_list = self.image_list[:image_num]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = os.path.join(self.images_path, self.image_list[idx])
        # image = np.array(Image.open(image_name).convert('RGB'))
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            # image = self.transform(image=image)['image']
            image = self.transform(image)
        return image


#
# train_transform = A.Compose([
#     # A.Resize(width=500, height=500),
#     A.RandomCrop(width=256, height=256),
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(always_apply=False, p=0.5),
#     # A.RandomRotate90(always_apply=False, p=0.5),
#     # A.RandomBrightnessContrast(p=0.2),
#     # A.ToGray(p=0.2),
#     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
#     ToTensorV2()
# ])
# test_transform = A.Compose([
#     # A.Resize(width=500, height=500),
#     A.CenterCrop(height=256, width=256),
#     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
#     ToTensorV2()
# ])
# basic_transform = A.Compose([
#     A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
#     ToTensorV2()
# ])

train_transform = transforms.Compose(
    [
        transforms.Resize(400),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize(400),
        transforms.CenterCrop(256),
        transforms.ToTensor()
    ]
)

from args_file import set_args
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image

if __name__ == "__main__":
    args = set_args()
    image_set = ImageDataset(
        images_path=args.image_path,
        transform=train_transform,
        image_num=10
    )
    image_loader = data.DataLoader(
        image_set,
        batch_size=1,
        shuffle=False
    )
    print(image_set.__len__())
    print(image_set.image_list)

    im = image_set.__getitem__(3)
    # plt.imshow(im)
    # plt.show()

    # print('=-' * 30)
    # print(type(im))
    # print(im.shape)
    # print(im.max())
    # print(im.min())

    for i, im in enumerate(image_loader):
        print('=-' * 30)
        print(type(im))
        print(im.shape)
        print(im.max())
        print(im.min())
        print()
        save_image(im, './q{}.jpg'.format(i), normalize=False)
