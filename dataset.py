import os
import glob
import numpy as np
import imgaug as ia
import PIL.Image as Image
import imgaug.augmenters as iaa
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image


class MyDataset(Dataset):
    def __init__(self, root, style, mode=None, transform=None, augmentations=None):
        super(MyDataset, self).__init__()
        self.root = root
        self.style = style
        self.mode = mode
        self.augmentations = augmentations  

        if style == "edge_smoothed":
            self.dir = os.path.join(self.root, style)
            file_pattern = '*.png'
        else:

            self.dir = os.path.join(self.root, mode)
            file_pattern = '*.jpg'
        #self.path_list = glob.glob(os.path.join(self.dir, file_pattern))
        self.path_list = []
        for class_dir in glob.glob(os.path.join(self.dir, '*')):  # 遍历每个类的文件夹
            if os.path.isdir(class_dir):  # 确保是文件夹
                self.path_list.extend(glob.glob(os.path.join(class_dir, file_pattern)))

        self.transform = transform or transforms.Compose([
            #transforms.Resize((32, 32)),
            transforms.Resize((2048, 2048)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
            #三通道
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            #])

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path = self.path_list[idx]
        # 灰色选项
        image = Image.open(img_path).convert('L')
        #彩色选项
        #image = Image.open(img_path)
        
        if self.augmentations and self.mode == 'train':
            image = np.array(image)
            augmented_image = self.augmentations(image)  
            
            if isinstance(augmented_image, list):
                augmented_image = augmented_image[0]
            
            image = to_pil_image(augmented_image)  

        if self.transform:
            image = self.transform(image)
        return image


class Augment(object):
    def __init__(self):
        def sometimes(aug): return iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential(
            [
                iaa.Fliplr(0.5), 
                iaa.Flipud(0.2), 
                sometimes(iaa.Crop(px=(1, 16), keep_size=False)),
                sometimes(iaa.Affine(
                    scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                    translate_percent={
                        "x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                    rotate=(-45, 45),
                    shear=(-16, 16), 
                    order=[0, 1],
                    cval=(0, 255),
                    mode=ia.ALL
                )),
                iaa.Resize({"height": 32, "width": 32}) 
            ],
            random_order=False
        )

    def __call__(self, img):
        augmented_image = self.seq(images=img.astype(np.uint8))
        return augmented_image