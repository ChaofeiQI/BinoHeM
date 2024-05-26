from __future__ import absolute_import
from __future__ import division
import torch, random, math
from PIL import Image
import numpy as np
from torchvision.transforms import *
from abc import abstractmethod
import torchvision.transforms.functional as F
from data.dataset import SimpleDataset, SetDataset, EpisodicBatchSampler

class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
    - height (int): target height.
    - width (int): target width.
    - p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
        - img (PIL Image): Image to be cropped.
        """
        if random.uniform(0, 1) > self.p:
            return img.resize((self.width, self.height), self.interpolation)
        
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value. 
    """
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                return img

        return img

class TransformLoader:
    def __init__(self, image_size, dataset):
        self.normalize_param = dict(mean=[0.0, 0.0, 0.0], std=[0.0, 0.0, 0.0])
        self.image_size = image_size
        self.dataset = dataset

    def get_composed_transform(self, aug=False, height=84, width=84):
        # 数据集1：cub
        if self.dataset == 'cub_cropped':
            self.normalize_param = dict(mean=[0.4723, 0.4714, 0.4057], std=[0.229, 0.224, 0.225])
            if aug: transform = Compose([
                    Resize((height, width), interpolation=3),
                    RandomCrop(height, padding=8),
                    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(**self.normalize_param),
                    RandomErasing(0.5) 
                    ])
            else:   transform = Compose([
                    Resize((height, width), interpolation=3),
                    ToTensor(),
                    Normalize(**self.normalize_param),
                    ])
        # 数据集3：aircraft_fs
        elif self.dataset == 'aircraft_fs':
            self.normalize_param = dict(mean=[0.4901, 0.5137, 0.5358], std=[0.2162, 0.2116, 0.2245])
            if aug: transform = Compose([
                    Resize((height, width), interpolation=3),
                    RandomCrop((height, width), padding=8),
                    ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(**self.normalize_param),
                    RandomErasing(0.5) 
                    ])
            else:   transform = Compose([
                    Resize((height, width), interpolation=3),
                    ToTensor(),
                    Normalize(**self.normalize_param),
                    ])
        # 其他数据集：
        elif self.dataset == 'flowers_fs' or self.dataset == 'stanford_car':
            self.normalize_param = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
            if aug: transform = Compose([
                    Resize((height, width), interpolation=3),
                    RandomCrop(height, padding=8),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(**self.normalize_param),
                    RandomErasing(0.5)
                    ])
            else:   transform = Compose([
                    Resize((height, width), interpolation=3),
                    ToTensor(),
                    Normalize(**self.normalize_param), 
                    ])
        return transform


#############################
# 1.预训练
#############################
class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass

class SimpleDataManager(DataManager):
    def __init__(self, data_path, image_size, batch_size, dataset):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.trans_loader = TransformLoader(image_size, dataset)
        self.dataset = dataset

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SimpleDataset(self.data_path, data_file, transform)
        # data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=12, pin_memory=True)
        data_loader_params = dict(batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader

#############################
# 2.元训练
#############################
class SetDataManager(DataManager):
    def __init__(self, data_path, image_size, n_way, n_support, n_query, n_episode, dataset):
        super(SetDataManager, self).__init__()
        self.image_size = image_size
        self.n_way = n_way
        self.batch_size = n_support + n_query
        self.n_episode = n_episode
        self.data_path = data_path
        self.dataset = dataset
        self.trans_loader = TransformLoader(image_size, dataset)

    def get_data_loader(self, data_file, aug):  # parameters that would change on train/val set
        transform = self.trans_loader.get_composed_transform(aug)
        dataset = SetDataset(self.data_path, data_file, self.batch_size, transform)
        sampler = EpisodicBatchSampler(len(dataset), self.n_way, self.n_episode)
        # data_loader_params = dict(batch_sampler=sampler, num_workers=12, pin_memory=True)
        data_loader_params = dict(batch_sampler=sampler, num_workers=8, pin_memory=True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)
        return data_loader
