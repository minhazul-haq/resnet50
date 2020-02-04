#author: Mohammad Minhazul Haq
#created on: February 3, 2020

import numpy as np
from torch.utils import data
from PIL import Image
from skimage.io import imread
import os


class WSI_Dataset(data.Dataset):
    def __init__(self, dir, transform=None, mean=0.0, std=1.0):
        filenames = sorted(os.listdir(dir))
        filepaths = []

        for filename in filenames:
            filepaths.append(os.path.join(dir, filename))

        self.files = []

        for filepath in filepaths:
            filename = filepath.split('/')[-1]
            image_label = (filename.split('_')[-1]).replace(".jpg", "")

            self.files.append({"image": filepath,
                               "label": int(image_label),
                               "name": filename})

        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = imread(datafiles["image"])
        #image = Image.open(datafiles["image"])
        #image = np.array(image).astype('float32')
        #image = ((image - self.mean) / self.std)
        #image = image.transpose((2, 0, 1))

        if self.transform:
            image = self.transform(image)

        return image, datafiles["label"], datafiles["name"]
