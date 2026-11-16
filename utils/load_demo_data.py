import numpy as np
import torch.utils.data as data
from os.path import join
from PIL import Image
import random
from random import randrange
import torch
import imageio
from skimage.transform import resize
import os

def generate_4masks(mask):
    mask_pil = Image.fromarray(mask)

    (width2, height2) = (mask_pil.width // 2, mask_pil.height // 2)
    (width3, height3) = (mask_pil.width // 4, mask_pil.height // 4)
    (width4, height4) = (mask_pil.width // 8, mask_pil.height // 8)

    mask2 = mask_pil.resize((width2, height2))
    mask3 = mask_pil.resize((width3, height3))
    mask4 = mask_pil.resize((width4, height4))

    mask = mask.astype(np.float32) / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    mask2 = np.asarray(mask2).astype(np.float32) / 255
    mask2[mask2 > 0.5] = 1
    mask2[mask2 <= 0.5] = 0

    mask3 = np.asarray(mask3).astype(np.float32) / 255
    mask3[mask3 > 0.5] = 1
    mask3[mask3 <= 0.5] = 0

    mask4 = np.asarray(mask4).astype(np.float32) / 255
    mask4[mask4 > 0.5] = 1
    mask4[mask4 <= 0.5] = 0

    mask = torch.from_numpy(mask)
    mask2 = torch.from_numpy(mask2)
    mask3 = torch.from_numpy(mask3)
    mask4 = torch.from_numpy(mask4)

    return mask, mask2, mask3, mask4


def data_aug(img, data_aug_ind):
    img = Image.fromarray(img)
    if data_aug_ind == 0:
        return np.asarray(img)
    elif data_aug_ind == 1:
        return np.asarray(img.rotate(90, expand=True))
    elif data_aug_ind == 2:
        return np.asarray(img.rotate(180, expand=True))
    elif data_aug_ind == 3:
        return np.asarray(img.rotate(270, expand=True))
    elif data_aug_ind == 4:
        return np.asarray(img.transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 5:
        return np.asarray(img.rotate(90, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 6:
        return np.asarray(img.rotate(180, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    elif data_aug_ind == 7:
        return np.asarray(img.rotate(270, expand=True).transpose(Image.FLIP_TOP_BOTTOM))
    else:
        raise Exception('Data augmentation index is not applicable.')

class YD_TrainDataset(data.Dataset):
    def __init__(self, args):
        super(YD_TrainDataset, self).__init__()
        path, crop_size, train_num, train_ratio, val_num = args['path'], args['crop_size'], args['train_num'], args['train_ratio'], args['val_num']

        auth_dir = '/home/v-sunanhe/blob2/v-sunanhe/1_Research_Project/FaceShifter/Yidong_Origin'
        names = os.listdir(auth_dir) # [:10]
        authentic_names = [os.path.join(auth_dir, name) for name in names]
        authentic_class = [0] * len(authentic_names)

        fake_dir = "/home/v-sunanhe/blob2/v-sunanhe/1_Research_Project/FaceShifter/Yidong_Swap"
        names = os.listdir(fake_dir) 
        fake_names = [os.path.join(fake_dir, name) for name in names if "img" in name]
        fake_class = [2] * len(fake_names)

        self.image_names = authentic_names + fake_names
        self.image_class = authentic_class + fake_class

        # self.image_names = [authentic_names, fake_names]
        # self.image_class = [authentic_class, fake_class]

        self.train_num = len(self.image_names)
        self.train_ratio = train_ratio
        self.crop_size = crop_size

    def rgba2rgb(self, rgba, background=(255, 255, 255)):
        row, col, ch = rgba.shape

        rgb = np.zeros((row, col, 3), dtype='float32')
        r, g, b, a = rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], rgba[:, :, 3]

        a = np.asarray(a, dtype='float32') / 255.0

        R, G, B = background

        rgb[:, :, 0] = r * a + (1.0 - a) * R
        rgb[:, :, 1] = g * a + (1.0 - a) * G
        rgb[:, :, 2] = b * a + (1.0 - a) * B

        return np.asarray(rgb, dtype='uint8')

    def get_item(self, index):

        crop_width, crop_height = self.crop_size
        train_num = self.train_num
        train_ratio = self.train_ratio

        # read the chosen image
        image_name = self.image_names[index]
        image = imageio.imread(image_name)
        cls = self.image_class[index]
        im_height, im_width, im_channel = image.shape

        if im_channel != 3:
            print(image_name)
            raise Exception('Image channel is not 3.')

        # authentic
        if cls == 0:
            if image.shape[-1] == 4:
                image = self.rgba2rgb(image)

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(np.uint8(image))
                # image = Image.fromarray(image.astype(np.uint8))
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)

            mask = np.zeros((crop_height, crop_width)).astype(np.uint8)

        # splice
        elif cls == 2:
            if '.jpg' in image_name:
                mask_name = image_name.replace('img', 'msk')
            else:
                mask_name = image_name.replace('img', 'msk')

            mask = imageio.imread(mask_name)
            # mask = resize(mask, (1024, 1024))
            ma_height, ma_width = mask.shape[:2]

            if im_width != ma_width or im_height != ma_height:
                raise Exception('the sizes of image and mask are different: {}'.format(image_name))

            if im_height != crop_height or im_width != crop_width:
                # resize image
                image = Image.fromarray(np.uint8(image))

                # image = Image.fromarray(image)
                image = image.resize((crop_height, crop_width), resample=Image.BICUBIC)
                image = np.asarray(image)
                # resize mask
                mask = Image.fromarray(np.uint8(mask))
                # mask = Image.fromarray(mask)
                mask = mask.resize((crop_height, crop_width), resample=Image.BICUBIC)
                mask = np.asarray(mask)
                if mask.sum() == 0:
                    raise NameError

        return image, mask, cls

    def __getitem__(self, index):
        res = self.get_item(index)
        return res

    def __len__(self):
        return self.train_num
