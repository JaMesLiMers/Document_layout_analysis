import os

import cv2
import numpy as np
from augmentation import Augmentation
from torch.utils.data import Dataset

class ModelDataset(Dataset):
    def __init__(self, base_path='./Data', train=True):
        super(ModelDataset, self).__init__()
        # parameters
        # TODO add config file
        self.BLUR = 0.0
        self.FLIP = 0.0
        self.COLOR = 0.0
        self.GRAY = 0.2
        self.SIZE = (400, 600)
        self.train = train

        # augumentation
        self.augumentation = Augmentation(self.BLUR, self.FLIP, self.COLOR)

        # build path
        self.base_path = base_path
        self.image_path = os.path.join(base_path, 'Image')
        self.anno_path = os.path.join(base_path, 'Annotations')

        # build list
        self.image_list = self.file_name(self.image_path, '.jpg')
        self.image_list.sort()

        if self.train:
            self.mask_list = self.file_name(self.anno_path, '.jpg')
            self.mask_list.sort()
            self.csv_list = self.file_name(self.anno_path, '.csv')
            self.csv_list.sort()

            # check all have label
            for i in self.image_list[:]:
                if i not in self.mask_list:
                    print("{} don't have annotation!".format(i))
                    self.image_list.remove(i)

        # number of data
        self.num = len(self.image_list)

    def __getitem__(self, index):
        # init
        target_mask = None
        # make target choice
        # target = np.random.choice(self.image_list)
        target = self.image_list[index]
        # read data
        target_image = cv2.imread(os.path.join(self.image_path, target))
        if self.train:
            # TODO add csv reader
            target_mask = cv2.imread(os.path.join(self.anno_path, target))

            # creat label
            gray = self.GRAY and self.GRAY > np.random.random()

            target_image, target_mask =  self.augumentation(target_image, 
                                                            target_mask,
                                                            self.SIZE,
                                                            gray=gray)

            # compile
            target_mask = np.where(target_mask, 1, 0)
            blank_mask = np.where(np.sum(target_mask, axis=2)==0, 1, 0)[:,:,np.newaxis]
            target_mask = np.dstack((blank_mask, target_mask))
            target_mask = target_mask.transpose((2, 1, 0)).astype(np.float32)
        else:
            target_image = cv2.resize(target_image, self.SIZE)

        target_image = target_image.transpose((2, 1, 0)).astype(np.float32)

        return {
                'target_image': target_image,
                'target_mask' : target_mask,
                } 

    def __len__(self):
        return self.num

    def file_name(self, file_dir, target='.jpg'):
        File_Name=[]
        for files in os.listdir(file_dir):
            if os.path.splitext(files)[1] == target:
                File_Name.append(files)
        return File_Name


if __name__ == "__main__":
    import torch
    # 准备数据集
    train_set = ModelDataset(base_path='./Data', train=True)

    # 建立dataloader
    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=10,
                    num_workers=1,
                    shuffle=True)

    for i in train_loader:
        print(i)


